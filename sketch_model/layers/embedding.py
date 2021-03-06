from typing import Tuple, Optional

import torch
from torch import nn

from sketch_model.configs import SketchModelConfig
from sketch_model.configs.config import SentenceMethod, Aggregation, PosPattern
from sketch_model.utils import NestedTensor


class TextEmbedding(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            vocab_size: int,
            pad_token_id: int,
            dropout_rate: float,
            sentence_method: SentenceMethod = SentenceMethod.SUM,
            layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim,
                                            padding_idx=pad_token_id)
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_rate)
        self.sentence_method = sentence_method

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        if self.sentence_method == SentenceMethod.SUM:
            return embeddings.sum(dim=2)
        elif self.sentence_method == SentenceMethod.MAX:
            return embeddings.max(dim=2)[0]
        elif self.sentence_method == SentenceMethod.MEAN:
            return embeddings.mean(dim=2)


class ImageEmbedding(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            patch_size: int,
            image_size: int = 64,
            num_channels: int = 3
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.image_size = (image_size, image_size)
        self.num_channels = num_channels
        self.projection = nn.Sequential(
            nn.Conv2d(num_channels, embedding_dim, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(),
            nn.Flatten(2),
            nn.Linear((image_size // self.patch_size) ** 2, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # batch_size, seq_len, channels, height, width = images.shape
        flat_images = images.view(-1, self.num_channels, *self.image_size)
        features = self.projection(flat_images)
        expand_features = features.view(*images.shape[:2], self.embedding_dim)
        return expand_features


class LayerStructureEmbedding(nn.Module):
    def __init__(
            self,
            config: SketchModelConfig,
    ):
        super().__init__()
        self.config = config

        self.freq_fc: Optional[nn.Module] = None
        self.coord_embeder: Optional[nn.Module] = None
        self.init_pos_embedding()

        self.image_embeder: Optional[nn.Module] = None
        self.init_image_embedding()

        self.token_embeder: Optional[nn.Module] = None
        self.init_token_embedding()

        self.color_embeder: Optional[nn.Module] = None  # RGBA
        self.init_color_embedding()

        self.class_embeder: Optional[nn.Module] = None  # Layer class
        self.init_class_embedding()

        self.concat_embeder: Optional[nn.Module] = None
        self.init_concat_embedding()

    @property
    def embedding_dim(self):
        return self.config.hidden_dim

    def init_pos_embedding(self):
        if self.config.pos_pattern == PosPattern.ONE:
            num_groups = 4
        elif self.config.pos_pattern == PosPattern.FOUR:
            num_groups = 1
        elif self.config.pos_pattern == PosPattern.TWO:
            num_groups = 2
        else:
            raise ValueError("Unknown pos pattern")
        self.freq_fc = nn.Linear(4 // num_groups, self.config.pos_freq)
        self.coord_embeder = nn.Sequential(
            nn.Linear(self.config.pos_freq * 2, self.embedding_dim // num_groups),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // num_groups, self.embedding_dim // num_groups),
        )

    def init_token_embedding(self):
        if self.config.use_name:
            self.token_embeder = TextEmbedding(
                self.embedding_dim,
                self.config.vocab_size,
                self.config.pad_token_id,
                self.config.dropout,
                self.config.sentence_method
            )

    def init_image_embedding(self):
        if self.config.use_image:
            self.image_embeder = ImageEmbedding(self.embedding_dim, 4)

    def init_color_embedding(self):
        if self.config.use_color:
            self.color_embeder = nn.Linear(4, self.embedding_dim)

    def init_class_embedding(self):
        if self.config.use_class:
            self.class_embeder = nn.Embedding(self.config.class_types, self.embedding_dim)

    def init_concat_embedding(self):
        if self.config.aggregation == Aggregation.CONCAT:
            count = 0
            if self.config.use_image:
                count += 1
            if self.config.use_name:
                count += 1
            if self.config.use_color:
                count += 1
            if self.config.use_class:
                count += 1

            self.concat_embeder = nn.Linear(self.embedding_dim * count, self.embedding_dim)

    def forward(
            self,
            images: NestedTensor,
            names: NestedTensor,
            boxes: NestedTensor,
            colors: NestedTensor,
            classes: NestedTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeds = []
        if self.config.use_image:
            embeds.append(self.image_embeder(images.tensors))
        if self.config.use_name:
            embeds.append(self.token_embeder(names.tensors))
        if self.config.use_color:
            embeds.append(self.color_embeder(colors.tensors))
        if self.config.use_class:
            embeds.append(self.class_embeder(classes.tensors))
        if len(embeds) == 0:
            raise ValueError("No embedding")
        if self.config.aggregation == Aggregation.CONCAT:
            embeds = torch.cat(embeds, dim=-1)
            embeds = self.concat_embeder(embeds)
        else:
            embeds = torch.sum(torch.stack(embeds, dim=-1), dim=-1)
        pos_embeds = self.pos_embeder(boxes.tensors)

        embeds = nn.Dropout(self.config.dropout)(embeds)
        pos_embeds = nn.Dropout(self.config.dropout)(pos_embeds)

        return embeds, pos_embeds

    def pos_embeder(
            self,
            boxes: torch.Tensor
    ) -> torch.Tensor:
        if self.config.pos_pattern == PosPattern.ONE:
            boxes = torch.unsqueeze(boxes, dim=3)
        elif self.config.pos_pattern == PosPattern.FOUR:
            boxes = torch.unsqueeze(boxes, dim=2)
        elif self.config.pos_pattern == PosPattern.TWO:
            boxes = torch.reshape(boxes, boxes.shape[:2] + (2, 2))
        else:
            raise ValueError('Unknown pos pattern')
        freqs = self.freq_fc(boxes)
        features = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)
        coord_embeds: torch.Tensor = self.coord_embeder(features)
        coord_embeds = torch.reshape(coord_embeds, coord_embeds.shape[:2] + (-1,))
        return coord_embeds
