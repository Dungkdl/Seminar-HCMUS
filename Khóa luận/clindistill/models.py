from pathlib import Path

import torch
from torch import nn
from torchvision import models


class VisualBagClassifier(nn.Module):
    def __init__(self, config, num_classes=6, load_pretrained=True):
        super().__init__()
        self.freeze_backbone = config.freeze_backbone
        self.pooling = 'mean' if config.experiment in {'single_photo', 'photo_mean'} else 'attention'
        if config.experiment == 'photo_set_attention':
            self.pooling = 'set'
        if config.backbone == 'tiny_debug':
            self.encoder = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
                                         nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                                         nn.AdaptiveAvgPool2d(1), nn.Flatten())
            width = 32
        else:
            builder, enum, head_name = {
                'resnet18': (models.resnet18, models.ResNet18_Weights, 'fc'),
                'resnet50': (models.resnet50, models.ResNet50_Weights, 'fc'),
                'swin_b': (models.swin_b, models.Swin_B_Weights, 'head'),
            }[config.backbone]
            weights = enum.DEFAULT if config.pretrained and load_pretrained and not config.backbone_weights_path else None
            try:
                self.encoder = builder(weights=weights)
            except Exception as exc:
                raise RuntimeError('Cannot load pretrained weights. On offline Kaggle, attach the official state_dict '
                                   'and set backbone_weights_path; random initialization is not substituted.') from exc
            if config.backbone_weights_path and load_pretrained:
                state = torch.load(Path(config.backbone_weights_path), map_location='cpu', weights_only=True)
                self.encoder.load_state_dict(state, strict=True)
            width = getattr(self.encoder, head_name).in_features
            setattr(self.encoder, head_name, nn.Identity())
        if self.freeze_backbone:
            self.encoder.requires_grad_(False)
        self.projection = nn.Sequential(nn.Linear(width, config.feature_dim), nn.ReLU())
        self.modality_embedding = nn.Embedding(2, config.feature_dim)
        # Initialize modality offsets to zero so photo-only baselines start equivalently.
        nn.init.zeros_(self.modality_embedding.weight)
        self.attention = nn.Sequential(nn.Linear(config.feature_dim, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(config.feature_dim, num_classes))
        if self.pooling == 'set':
            self.set_blocks = nn.ModuleList([SetAttentionBlock(config.feature_dim, config.dropout) for _ in range(2)])
            self.pool_query = nn.Parameter(torch.zeros(1, 1, config.feature_dim))
            nn.init.normal_(self.pool_query, std=0.02)
            self.set_pool = nn.MultiheadAttention(config.feature_dim, 4, batch_first=True)
            self.set_norm = nn.LayerNorm(config.feature_dim)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.eval()  # Frozen BatchNorm must not update statistics.
        return self

    def forward(self, images, mask, modalities):
        if not mask.any(dim=1).all():
            raise ValueError('Every bag must contain a valid image.')
        # Padding never reaches the visual encoder or its BatchNorm.
        valid_features = self.projection(self.encoder(images[mask]))
        # Embeddings can promote autocast features to float32; allocate after the addition.
        valid_features = valid_features + self.modality_embedding(modalities[mask])
        features = valid_features.new_zeros((*mask.shape, valid_features.shape[-1]))
        features[mask] = valid_features
        if self.pooling == 'set':
            for block in self.set_blocks:
                features = block(features, mask)
            pooled, attention = self.set_pool(self.pool_query.expand(images.shape[0], -1, -1),
                                              features, features, key_padding_mask=~mask)
            pooled, attention = self.set_norm(pooled[:, 0]), attention[:, 0]
            return {'logits': self.head(pooled), 'attention': attention, 'features': pooled}
        if self.pooling == 'attention':
            scores = self.attention(features).squeeze(-1).masked_fill(~mask, float('-inf'))
            attention = torch.softmax(scores, dim=1)
        else:
            attention = mask.to(features.dtype) / mask.sum(dim=1, keepdim=True)
        pooled = (features * attention.unsqueeze(-1)).sum(dim=1)
        return {'logits': self.head(pooled), 'attention': attention, 'features': pooled}


class SetAttentionBlock(nn.Module):
    """Self-attention without positional encoding; masked keys and row-wise feed-forward."""
    def __init__(self, width, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, 4, dropout=dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(width * 2, width))

    def forward(self, features, mask):
        attended, _ = self.attention(features, features, features, key_padding_mask=~mask, need_weights=False)
        features = self.norm1(features + attended)
        return self.norm2(features + self.ff(features))
