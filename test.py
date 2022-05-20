import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from main import init_device, init_config, init_model
from sketch_model import utils
from sketch_model.configs import SketchModelConfig, config_with_arg
from sketch_model.datasets import build_dataset
from sketch_model.utils import NestedTensor, accuracy


class GroupDrawer:
    draw: ImageDraw.ImageDraw
    tmp_group: List[Tuple[float, float]]
    all_groups: List[Tuple[float, float, float, float]]

    def __init__(self, draw: ImageDraw.ImageDraw):
        self.draw = draw
        self.tmp_group = []
        self.all_groups = []

    def check(self, label, x1, y1, x2, y2):
        if len(self.tmp_group) > 0 and label != 3:
            np_stack = np.array(self.tmp_group)
            min_x = np.min(np_stack[:, 0])
            min_y = np.min(np_stack[:, 1])
            max_x = np.max(np_stack[:, 0])
            max_y = np.max(np_stack[:, 1])
            self.all_groups.append((min_x, min_y, max_x, max_y))
            self.tmp_group.clear()
        if label > 1:
            self.tmp_group.append((x1, y1))
            self.tmp_group.append((x2, y2))

    def draw_groups(self):
        for x1, y1, x2, y2 in self.all_groups:
            self.draw.rectangle((x1, y1, x2, y2), outline='red')


def draw_labeled_artboard(
        artboard_detail: Dict[str, Any],
        image_assets: List[np.ndarray],
        pred: np.ndarray,
        target: np.ndarray,
        show: bool = False,
):
    width = artboard_detail["width"]
    height = artboard_detail["height"]
    layers = artboard_detail["layers"]
    canvas = Image.new("RGBA", (2 * width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    pred_drawer = GroupDrawer(draw)
    target_drawer = GroupDrawer(draw)
    for layer, image, pred, target in zip(layers, image_assets, pred, target):
        x, y, w, h = layer["rect"].values()
        if w > 0 and h > 0:
            layer_image = Image.fromarray(image, "RGBA").resize((w, h))
            canvas.alpha_composite(layer_image, (x, y))
            canvas.alpha_composite(layer_image, (x + width, y))
        pred_drawer.check(pred, x, y, x + w, y + h)
        target_drawer.check(target, x + width, y, x + width + w, y + h)
    pred_drawer.draw_groups()
    target_drawer.draw_groups()
    canvas.show() if show else None


@torch.no_grad()
def get_res(
        input_data: Tuple[
            Tuple[NestedTensor, NestedTensor, NestedTensor, NestedTensor, NestedTensor],
            List[torch.Tensor]
        ],
        model: nn.Module,
        device: torch.device,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    (batch_img,
     batch_name,
     batch_bbox,
     batch_color,
     batch_class), targets = input_data
    batch_img = batch_img.to(device)
    batch_name = batch_name.to(device)
    batch_bbox = batch_bbox.to(device)
    batch_color = batch_color.to(device)
    batch_class = batch_class.to(device)
    targets = [t.to(device) for t in targets]

    outputs = model(batch_img, batch_name, batch_bbox,
                    batch_color, batch_class)
    return [
        (output[:len(target)].max(-1)[1].detach().numpy(), target.detach().numpy())
        for output, target
        in zip(outputs, targets)
    ]


def load_multiple_artboard(index_json_path: str, indexes: List[int]) -> Tuple[str, str]:
    index_json = json.load(open(index_json_path))
    index_artboards = [index_json[i] for i in indexes]
    tmp_path = tempfile.mkdtemp()
    new_json_path = os.path.join(tmp_path, f"temp.json")
    with open(new_json_path, "w") as f:
        json.dump(index_artboards, f)
    return tmp_path, new_json_path


def main(config: SketchModelConfig, indexes: List[int]):
    device = init_device(config)
    config, checkpoint = init_config(config)
    tokenizer, model, criterion, optimizer, lr_scheduler = init_model(config, checkpoint, device)
    tmp_path, new_json_path = load_multiple_artboard(config.test_index_json, indexes)
    dataset_val = build_dataset(new_json_path, Path(config.test_index_json).parent.__str__(), tokenizer)
    sampler_val = SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, config.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=config.num_workers)
    model.eval()
    for batch_index, x in tqdm(enumerate(data_loader_val)):
        res = get_res(x, model, device)
        for index_in_batch, (pred, target) in enumerate(res):
            acc = accuracy(pred, target)
            if acc > 80 and (2 in pred or 3 in pred):
                real_index = batch_index * config.batch_size + index_in_batch
                artboard_index = dataset_val.index_json[real_index]
                artboard_detail = dataset_val.artboard_detail[real_index]
                print(f"{real_index} {artboard_index} {acc}")
                single_height = artboard_detail['layer_height']
                image_assets = np.asarray(dataset_val.load_image(artboard_index))
                image_assets = [
                    image_assets[i * single_height:(i + 1) * single_height]
                    for i in range(int(image_assets.shape[0] / single_height))
                ]
                draw_labeled_artboard(artboard_detail, image_assets, pred, target, True)
    shutil.rmtree(tmp_path)


if __name__ == '__main__':
    indexes = list(range(1000))
    main(config_with_arg(), indexes)
