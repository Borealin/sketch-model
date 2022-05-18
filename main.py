import datetime
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer

from sketch_model.configs import SketchModelConfig, get_config
from sketch_model.datasets import build_dataset
from sketch_model.model import build
from sketch_model.utils import misc as utils


def main(config: SketchModelConfig):
    device = torch.device(config.device)

    # fix the seed for reproducibility
    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("Loading Tokenizer...")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    tokenizer.model_max_length = config.max_name_length

    model, criterion = build(config, tokenizer)
    model.to(device)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=config.lr,
                                  weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)

    print("Loading Dataset...")
    dataset_train = build_dataset(config.train_index_json, tokenizer)
    dataset_val = build_dataset(config.test_index_json, tokenizer)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, config.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=config.num_workers)
    data_loader_val = DataLoader(dataset_val, config.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=config.num_workers)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    if config.resume:
        checkpoint = torch.load(config.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.start_epoch = checkpoint['epoch'] + 1

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if config.evaluate:
        test_stats = evaluate(model, criterion, data_loader_val, device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(config.start_epoch, config.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            config.clip_max_norm)
        lr_scheduler.step()
        if config.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 10 epochs
            if (epoch + 1) % config.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'config': config,
                }, checkpoint_path)
        test_stats = evaluate(model, criterion, data_loader_val, device)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if config.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        max_norm: float = 0,
):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
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
        outputs = model(batch_img, batch_name, batch_bbox, batch_color, batch_class)
        batch_ce_loss = torch.tensor(0.0, device=device)
        for i in range(len(targets)):
            ce_loss = criterion(outputs[i][:len(targets[i])], targets[i])
            batch_ce_loss += ce_loss

        if not math.isfinite(batch_ce_loss):
            print("Loss is {}, stopping training".format(batch_ce_loss))
            sys.exit(1)

        optimizer.zero_grad()
        batch_ce_loss.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=batch_ce_loss.detach())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, dataloader, device):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
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

        outputs = model(batch_img, batch_name, batch_bbox, batch_color, batch_class)
        batch_ce_loss = torch.tensor(0.0, device=device)
        for i in range(len(targets)):
            ce_loss = criterion(outputs[i][:len(targets[i])], targets[i])
            batch_ce_loss += ce_loss

        metric_logger.update(loss=batch_ce_loss.detach())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


if __name__ == '__main__':
    main(get_config())
