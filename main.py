import datetime
import json
import math
import os
import random
import sys
import time
import urllib.parse
from pathlib import Path

import numpy
import numpy as np
import requests
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, PreTrainedTokenizer

from sketch_model.configs import SketchModelConfig, config_with_arg
from sketch_model.datasets import build_dataset
from sketch_model.model import build
from sketch_model.utils import misc as utils, accuracy, f1score, r2score
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(config: SketchModelConfig):
    if config.device == 'cuda' and not torch.cuda.is_available():
        config.device = 'cpu'
    device = torch.device(config.device)
    # fix the seed for reproducibility
    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("Loading Tokenizer...")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name)
    tokenizer.model_max_length = config.max_name_length
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    model, criterion = build(config)
    model.to(device)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters(
        ) if "backbone" not in n and p.requires_grad]},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=config.lr,
                                  weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)

    print("Loading Dataset...")
    dataset_train = build_dataset(config.train_index_json, tokenizer)
    dataset_val = build_dataset(config.test_index_json, tokenizer)
    sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_val)
    batch_sampler_train = BatchSampler(
        sampler_train, config.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=config.num_workers)
    data_loader_val = DataLoader(dataset_val, config.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=config.num_workers)

    if config.resume:
        checkpoint = torch.load(config.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.start_epoch = checkpoint['epoch'] + 1

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if config.evaluate:
        test_stats = evaluate(config, model, criterion, data_loader_val, device)
        return

    output_dir = Path(config.output_dir) / config.task_name
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    tensorboard_dir = output_dir / 'tensorboard'
    os.makedirs(tensorboard_dir, exist_ok=True)
    config.save(output_dir / 'config.json')

    print("Start training")
    start_time = time.time()
    writer = SummaryWriter(str(tensorboard_dir))
    best_train_acc, best_train_f1, best_test_acc, best_test_f1 = [0] * 4
    for epoch in range(config.start_epoch, config.epochs):
        train_stats = train_one_epoch(config, model, criterion, data_loader_train, optimizer, device, epoch,
                                      config.clip_max_norm)
        lr_scheduler.step()
        if config.output_dir:
            checkpoint_paths = [checkpoint_dir / 'latest.pth']
            # extra checkpoint before LR drop and every 10 epochs
            if (epoch + 1) % config.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(
                    checkpoint_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'config': config,
                }, checkpoint_path)
        test_stats = evaluate(config, model, criterion, data_loader_val, device)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        best_train_acc, best_train_f1, best_test_acc, best_test_f1 = np.maximum(
            (best_train_acc, best_train_f1, best_test_acc, best_test_f1),
            (train_stats['acc'], train_stats['f1'], test_stats['acc'], test_stats['f1'])
        )
        writer.add_scalar('train/loss', train_stats['loss'], epoch)
        writer.add_scalar('train/acc', train_stats['acc'], epoch)
        writer.add_scalar('train/f1', train_stats['f1'], epoch)
        writer.add_scalar('train/r2', train_stats['r2'], epoch)
        writer.add_scalar('test/loss', test_stats['loss'], epoch)
        writer.add_scalar('test/acc', test_stats['acc'], epoch)
        writer.add_scalar('test/f1', test_stats['f1'], epoch)
        writer.add_scalar('test/r2', train_stats['r2'], epoch)
        writer.add_scalar(
            "learning_rate", optimizer.param_groups[0]['lr'], epoch)

        if config.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    writer.flush()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('Best test acc: {}'.format(best_test_acc))
    print('Best test f1: {}'.format(best_test_f1))
    print('Best train acc: {}'.format(best_train_acc))
    print('Best train f1: {}'.format(best_train_f1))
    sct_key = os.environ.get('SCT_KEY')
    if sct_key:
        title = f'{config.task_name} finished'
        content = f'Training {config.task_name} finished, total time:{total_time_str}, best test acc:{best_test_acc}, best test f1:{best_test_f1}, best train acc:{best_train_acc}, best train f1:{best_train_f1}'
        res = requests.get(
            f"https://sctapi.ftqq.com/{sct_key}.send?title={urllib.parse.quote_plus(title)}&desp={urllib.parse.quote_plus(content)}")
        print(res.text)


def train_one_epoch(
        config: SketchModelConfig,
        model: nn.Module,
        criterion: nn.Module,
        dataloader: DataLoader[str],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        max_norm: float = 0,
):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('f1', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('r2', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = f'Task:{config.task_name} Epoch: [{epoch}]'
    print_freq = 10

    for (batch_img,
         batch_name,
         batch_bbox,
         batch_color,
         batch_class), targets in metric_logger.log_every(dataloader, print_freq, header):
        batch_img = batch_img.to(device)
        batch_name = batch_name.to(device)
        batch_bbox = batch_bbox.to(device)
        batch_color = batch_color.to(device)
        batch_class = batch_class.to(device)
        targets = [t.to(device) for t in targets]
        outputs = model(batch_img, batch_name, batch_bbox,
                        batch_color, batch_class)
        batch_ce_loss = torch.tensor(0.0, device=device)
        acc, f1, r2 = 0, 0, 0
        for i in range(len(targets)):
            packed = outputs[i][:len(targets[i])]
            ce_loss = criterion(packed, targets[i])
            batch_ce_loss += ce_loss
            pred = packed.max(-1)[1]
            acc += accuracy(pred, targets[i])
            f1 += f1score(pred, targets[i])
            r2 += r2score(pred, targets[i])
        acc, f1, r2 = numpy.array([acc, f1, r2]) / len(targets)
        if not math.isfinite(batch_ce_loss):
            print("Loss is {}, stopping training".format(batch_ce_loss))
            sys.exit(1)

        optimizer.zero_grad()
        batch_ce_loss.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        metric_logger.update(acc=acc, f1=f1, r2=r2)
        metric_logger.update(loss=batch_ce_loss.detach())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
        config: SketchModelConfig,
        model: nn.Module,
        criterion: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('acc', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('f1', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('r2', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = f'Task:{config.task_name} Test:'
    print_freq = 10

    for (batch_img,
         batch_name,
         batch_bbox,
         batch_color,
         batch_class), targets in metric_logger.log_every(dataloader, print_freq, header):
        batch_img = batch_img.to(device)
        batch_name = batch_name.to(device)
        batch_bbox = batch_bbox.to(device)
        batch_color = batch_color.to(device)
        batch_class = batch_class.to(device)
        targets = [t.to(device) for t in targets]

        outputs = model(batch_img, batch_name, batch_bbox,
                        batch_color, batch_class)
        batch_ce_loss = torch.tensor(0.0, device=device)
        acc, f1, r2 = 0, 0, 0
        for i in range(len(targets)):
            packed = outputs[i][:len(targets[i])]
            ce_loss = criterion(packed, targets[i])
            batch_ce_loss += ce_loss
            pred = packed.max(-1)[1]
            acc += accuracy(pred, targets[i])
            f1 += f1score(pred, targets[i])
            r2 += r2score(pred, targets[i])
        acc, f1, r2 = numpy.array([acc, f1, r2]) / len(targets)

        metric_logger.update(acc=acc, f1=f1, r2=r2)
        metric_logger.update(loss=batch_ce_loss.detach())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


if __name__ == '__main__':
    main(config_with_arg())
