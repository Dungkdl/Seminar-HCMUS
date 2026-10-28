import torch
from torch import nn


class GatedMIL(nn.Module):
    def __init__(self, input_dim, width, dropout):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU())
        self.value = nn.Linear(width, width)
        self.gate = nn.Linear(width, width)
        self.score = nn.Linear(width, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, mask):
        if not mask.any(1).all():
            raise ValueError('Every visit needs at least one photo.')
        features = self.project(features.float())
        scores = self.score(torch.tanh(self.value(features)) * torch.sigmoid(self.gate(features))).squeeze(-1)
        weights = scores.masked_fill(~mask, -torch.inf).softmax(1)
        return self.dropout((features * weights.unsqueeze(-1)).sum(1)), weights


class VisitClassifier(nn.Module):
    def __init__(self, visual_dim, classes, *, width=256, dropout=0.2,
                 role='student', clinical_dim=0, fusion='gated', alignment_dim=0):
        super().__init__()
        if role not in {'student', 'visual_teacher', 'multimodal_teacher'}:
            raise ValueError('Unknown research model role.')
        if fusion not in {'gated', 'concat'}:
            raise ValueError('Unknown fusion.')
        self.role, self.fusion = role, fusion
        self.mil = GatedMIL(visual_dim, width, dropout)
        if role == 'multimodal_teacher':
            if clinical_dim < 1:
                raise ValueError('Multimodal Teacher needs clinical dimensions.')
            self.text = nn.Sequential(nn.Linear(clinical_dim + 5, width), nn.LayerNorm(width), nn.GELU())
            self.fusion_gate = nn.Linear(width * 2, width)
            self.fusion_norm = nn.LayerNorm(width)
            self.concat = nn.Linear(width * 2, width)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, classes))
        self.alignment = nn.Linear(width, alignment_dim) if alignment_dim else None

    def forward(self, visual, mask, *, clinical=None, demographics=None, clinical_present=None):
        if self.role != 'multimodal_teacher' and any(x is not None for x in (clinical, demographics, clinical_present)):
            raise ValueError('Visual-only model cannot receive clinical inputs.')
        pooled, attention = self.mil(visual, mask)
        if self.role == 'multimodal_teacher':
            if any(x is None for x in (clinical, demographics, clinical_present)):
                raise ValueError('Clinical values and missingness are required for multimodal Teacher.')
            present = clinical_present.float().unsqueeze(-1)
            text = self.text(torch.cat([clinical.float(), demographics.float()], -1)) * present
            if self.fusion == 'gated':
                gate = torch.sigmoid(self.fusion_gate(torch.cat([pooled, text], -1))) * present
                pooled = self.fusion_norm(pooled + gate * text)
            else:
                pooled = self.fusion_norm(self.concat(torch.cat([pooled, text], -1)))
        result = {'logits': self.head(pooled), 'features': pooled, 'attention': attention}
        if self.alignment is not None:
            result['alignment'] = self.alignment(pooled)
        return result
