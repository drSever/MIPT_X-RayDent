##################################################################
# Swin Transformer backbone для Detectron2
# Использует timm для загрузки весов Swin-B (ImageNet-21k pretrain)
# Регистрирует backbone в detectron2 registry
#
# Использует официальный API timm:
#   - set_input_size() для адаптации к произвольному размеру входа
#   - forward_intermediates() для извлечения feature maps в формате NCHW
##################################################################

import torch
import torch.nn as nn
from typing import Dict, List

from detectron2.modeling.backbone import Backbone, BACKBONE_REGISTRY, FPN
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.layers import ShapeSpec

try:
    import timm
except ImportError:
    raise ImportError("Установите timm: pip install timm")


class SwinBackbone(Backbone):
    """
    Swin Transformer backbone через timm.
    Поддерживает произвольный размер входа через set_input_size().

    Поддерживаемые модели:
        swin_tiny_patch4_window7_224   — ~28M params
        swin_small_patch4_window7_224  — ~50M params
        swin_base_patch4_window7_224   — ~88M params  
        swin_large_patch4_window7_224  — ~197M params
    """

    OUT_CHANNELS = {
        "swin_base_patch4_window7_224":  [128, 256, 512, 1024],
        "swin_small_patch4_window7_224": [96,  192, 384, 768],
        "swin_tiny_patch4_window7_224":  [96,  192, 384, 768],
        "swin_large_patch4_window7_224": [192, 384, 768, 1536],
    }

    def __init__(self, model_name: str, pretrained: bool,
                 out_features: List[str], freeze_at: int = 0):
        super().__init__()

        self.model_name = model_name
        self._out_feature_names = out_features

        # strict_img_size=False + always_partition=True позволяют
        # работать с произвольными размерами входа
        self.swin = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
            strict_img_size=False,
            dynamic_mask=True,      # пересчитывает attention mask динамически
        )
        # Примечание по pretrain:
        # swin_base_patch4_window7_224          — ImageNet-1k  (1.3M изображений)
        # swin_base_patch4_window7_224.ms_in22k — ImageNet-21k (14M изображений, лучше)

        channels = self.OUT_CHANNELS.get(model_name, [128, 256, 512, 1024])
        strides  = [4, 8, 16, 32]
        stage_names = ["stage0", "stage1", "stage2", "stage3"]

        self._out_feature_channels = {}
        self._out_feature_strides  = {}
        for name, ch, stride in zip(stage_names, channels, strides):
            self._out_feature_channels[name] = ch
            self._out_feature_strides[name]  = stride

        self._current_img_size = None

        if freeze_at > 0:
            self._freeze_stages(freeze_at)

        # Вызываем set_input_size сразу при инициализации (модель ещё на CPU).
        # Это пересчитывает attention masks и relative_position_bias_table
        # под нужный размер входа до переноса модели на GPU.
        # img_size берём из конфига через параметр init_img_size.
        # По умолчанию 640 — стандартный размер ортопантомограмм.

    def init_input_size(self, img_size: int):
        """Вызвать ОДИН РАЗ после создания backbone, ДО переноса на GPU."""
        h = w = img_size
        self.swin.set_input_size(img_size=(h, w))
        self._current_img_size = (h, w)
        # Включаем gradient checkpointing — снижает потребление VRAM ~вдвое
        # за счёт пересчёта активаций при backward вместо их хранения
        self.swin.set_grad_checkpointing(enable=True)
        print(f"+ Swin backbone инициализирован для img_size={img_size}x{img_size} (grad checkpointing ON)")

    def _freeze_stages(self, num_stages: int):
        for param in self.swin.patch_embed.parameters():
            param.requires_grad = False
        for i in range(min(num_stages, len(self.swin.layers))):
            for param in self.swin.layers[i].parameters():
                param.requires_grad = False
        print(f"+ Заморожено {num_stages} стадий Swin backbone")

    def _update_input_size(self, h: int, w: int):
        """Обновляет input_resolution в stages если размер изменился.
        Вызывается только если размер отличается от инициализированного.
        ВАЖНО: должен вызываться только на CPU (до переноса модели на GPU).
        """
        if self._current_img_size == (h, w):
            return
        # Если модель уже на GPU — пропускаем пересчёт (размер должен совпадать
        # с тем, что был задан в init_input_size)
        device = next(self.swin.parameters()).device
        if device.type != 'cpu':
            return
        self._current_img_size = (h, w)
        self.swin.set_input_size(img_size=(h, w))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = x.shape

        # Адаптируем модель к текущему размеру входа
        self._update_input_size(H, W)

        # forward_intermediates — официальный API timm для извлечения
        # промежуточных features в формате NCHW
        intermediates = self.swin.forward_intermediates(
            x,
            indices=list(range(len(self.swin.layers))),  # все 4 стадии
            intermediates_only=True,
            output_fmt='NCHW',
        )

        stage_names = ["stage0", "stage1", "stage2", "stage3"]
        result = {}
        for name, feat in zip(stage_names, intermediates):
            if name in self._out_feature_names:
                result[name] = feat

        return result

    def output_shape(self) -> Dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_feature_names
        }


@BACKBONE_REGISTRY.register()
def build_swin_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Строит Swin + FPN backbone и регистрирует его в detectron2.
    Вызывается через cfg.MODEL.BACKBONE.NAME = "build_swin_fpn_backbone"
    """
    model_name   = cfg.MODEL.SWIN.MODEL_NAME
    pretrained   = cfg.MODEL.SWIN.PRETRAINED
    freeze_at    = cfg.MODEL.BACKBONE.FREEZE_AT
    in_features  = ["stage0", "stage1", "stage2", "stage3"]
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS  # обычно 256

    bottom_up = SwinBackbone(
        model_name=model_name,
        pretrained=pretrained,
        out_features=in_features,
        freeze_at=freeze_at,
    )

    # Инициализируем размер входа ДО переноса на GPU
    # Берём из конфига MIN_SIZE_TEST как целевой размер
    init_size = cfg.INPUT.MIN_SIZE_TEST if cfg.INPUT.MIN_SIZE_TEST else 640
    if isinstance(init_size, (list, tuple)):
        init_size = init_size[0]
    bottom_up.init_input_size(init_size)

    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )

    return backbone
