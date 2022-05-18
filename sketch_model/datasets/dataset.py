import json
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

LAYER_CLASS_MAP = {
    'symbolMaster': 0,
    'group': 1,
    'oval': 2,
    'polygon': 3,
    'rectangle': 4,
    'shapePath': 5,
    'star': 6,
    'triangle': 7,
    'shapeGroup': 8,
    'text': 9,
    'symbolInstance': 10,
    'slice': 11,
    'MSImmutableHotspotLayer': 12,
    'bitmap': 13,
}


class SketchDataset(Dataset):
    def __init__(self, index_json_path: str, tokenizer: PreTrainedTokenizerBase):
        self.data_folder = Path(index_json_path).parent
        self.index_json = json.load(open(index_json_path, 'r'))
        self.img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.data = self.load_data(tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_data(self, tokenizer: PreTrainedTokenizerBase):
        data = []
        for artboard in tqdm(self.index_json, desc='Loading Artboards'):
            json_path = self.data_folder / artboard['json']
            json_data = json.load(open(json_path, 'r'))
            single_layer_size = (json_data['layer_width'], json_data['layer_height'])
            asset_image_path = str(self.data_folder / artboard['layerassets'])
            asset_image_rgb = Image.open(asset_image_path).convert('RGB')
            asset_image_tensor = self.img_transform(asset_image_rgb)
            images = torch.stack(asset_image_tensor.split(single_layer_size[1], dim=1))
            names = []
            bboxes = []
            colors = []
            classes = []
            labels = []
            for layer in json_data['layers']:
                layer_name = layer['name'].lower()
                names.append(tokenizer.encode(
                    layer_name,
                    add_special_tokens=False,
                    padding=PaddingStrategy.MAX_LENGTH,
                    truncation=True,
                ))
                x1, y1, width, height = layer['rect']['x'], layer['rect']['y'], layer['rect']['width'], layer['rect'][
                    'height']
                x2, y2 = x1 + width, y1 + height
                bboxes.append([x1, y1, x2, y2])
                colors.append([color / 255.0 for color in layer['color']])
                classes.append(LAYER_CLASS_MAP[layer['_class']])
                labels.append(layer['label'])
            names = torch.as_tensor(names, dtype=torch.int64)
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
            colors = torch.as_tensor(colors, dtype=torch.float32)
            classes = torch.as_tensor(classes, dtype=torch.int64)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            data.append((images, names, bboxes, colors, classes, labels))
        return data


__all__ = ['SketchDataset']
