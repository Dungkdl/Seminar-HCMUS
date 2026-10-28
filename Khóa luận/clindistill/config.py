from dataclasses import asdict, dataclass

from .artifacts import read_json


@dataclass(frozen=True)
class TrainConfig:
    experiment: str = 'photo_attention'
    backbone: str = 'resnet50'
    pretrained: bool = True
    backbone_weights_path: str | None = None
    freeze_backbone: bool = False
    image_size: int = 224
    max_photos: int = 8
    max_radiographs: int = 2
    feature_dim: int = 256
    dropout: float = 0.2
    batch_size: int = 4
    epochs: int = 30
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    patience: int = 7
    seed: int = 2026
    workers: int = 2
    amp: bool = True
    class_weighted_loss: bool = False
    debug_max_cases: int = 0
    bootstrap_repetitions: int = 1000
    method: str = 'supervised'
    temperature: float = 4.0
    kd_weight: float = 1.0
    v_weight: float = 3.0
    h_weight: float = 1.0
    field_dropout: float = 0.2
    accumulation_steps: int = 1
    feature_kd_weight: float = 0.0
    relation_kd_weight: float = 0.0

    def __post_init__(self):
        if self.experiment not in {'single_photo', 'photo_mean', 'photo_attention', 'photo_xray_attention', 'photo_set_attention'}:
            raise ValueError('Unknown visual experiment.')
        if self.method not in {'supervised', 'teacher_clinical', 'teacher_full', 'teacher_v', 'teacher_h',
                               'kd_clinical', 'kd_full', 'kd_equal', 'kd_selective'}:
            raise ValueError('Unknown training method.')
        if self.temperature <= 0 or self.kd_weight < 0 or min(self.v_weight, self.h_weight) <= 0:
            raise ValueError('Invalid KD temperature or weights.')
        if not 0 <= self.field_dropout < 1 or self.accumulation_steps < 1:
            raise ValueError('Invalid field dropout or gradient accumulation.')
        if min(self.feature_kd_weight, self.relation_kd_weight) < 0:
            raise ValueError('Representation KD weights must be nonnegative.')
        if (self.feature_kd_weight or self.relation_kd_weight) and self.method != 'kd_full':
            raise ValueError('Representation KD ablations currently require the full multimodal teacher.')
        if self.experiment == 'photo_set_attention' and self.feature_dim % 4:
            raise ValueError('Set attention requires feature_dim divisible by four heads.')
        if self.backbone not in {'resnet18', 'resnet50', 'swin_b', 'tiny_debug'}:
            raise ValueError('Unsupported backbone.')
        if min(self.image_size, self.max_photos, self.max_radiographs, self.feature_dim,
               self.batch_size, self.epochs, self.patience) <= 0:
            raise ValueError('Sizes, budgets, epochs and patience must be positive.')
        if self.workers < 0 or self.debug_max_cases < 0 or self.bootstrap_repetitions < 1:
            raise ValueError('Invalid worker/debug/bootstrap setting.')
        if not 0 <= self.dropout < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError('Invalid optimizer/dropout setting.')
        if self.backbone == 'tiny_debug' and (self.pretrained or self.backbone_weights_path):
            raise ValueError('tiny_debug is random-init smoke-test only; set pretrained=false.')

    @classmethod
    def load(cls, path):
        return cls(**read_json(path))

    def to_dict(self):
        return asdict(self)
