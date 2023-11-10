"""EfficientNet models modified with added film layers.

Mostly taken from:
https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py
"""
import copy
import math
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torchvision.models._api import WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.efficientnet import (
    EfficientNet_B3_Weights,
    MBConv,
    MBConvConfig,
    WeightsEnum,
    _MBConvConfig,
)
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.utils import _log_api_usage_once

from robotic_transformer_pytorch.film_efficientnet import FilmConditioning


class MBConvFilm(nn.Module):
    """MBConv or FusedMBConv with FiLM context"""

    def __init__(self, mbconv: MBConv):
        super().__init__()
        self.mbconv = mbconv
        self.film = FilmConditioning(mbconv.block[-1][-1].num_features)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.mbconv(x)
        x = self.film(x, context)
        return x


class FilmEfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[MBConvConfig],
        dropout: float,
        include_top: bool = False,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = None,
    ) -> None:
        """
        EfficientNet V1 and V2 main class

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            include_top (bool): Whether to include the classification head
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        super().__init__()
        _log_api_usage_once(self)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError(
                "The inverted_residual_setting should be List[MBConvConfig]"
            )

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                )
                stage.append(
                    MBConvFilm(block_cnf.block(block_cnf, sd_prob, norm_layer))
                )
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = (
            last_channel if last_channel is not None else 4 * lastconv_input_channels
        )
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if include_top:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(lastconv_output_channels, num_classes),
            )
        else:
            self.classifier = nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if context is None:
            context = torch.zeros(x.shape[0], 512)
        for feature in self.features:
            for layer in feature:
                if isinstance(layer, MBConvFilm):
                    x = layer(x, context)
                else:
                    x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x


def _filmefficientnet(
    inverted_residual_setting: Sequence[MBConvConfig],
    dropout: float,
    last_channel: Optional[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> FilmEfficientNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = FilmEfficientNet(
        inverted_residual_setting, dropout, last_channel=last_channel, **kwargs
    )

    if weights is not None:
        state_dict = weights.get_state_dict(progress=progress)
        new_state_dict = {}
        for k, v in state_dict.items():
            if ".block" in k:
                new_state_dict[k.replace(".block", ".mbconv.block")] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(
            new_state_dict,
            strict=False,
        )

    return model


def _filmefficientnet_conf(
    arch: str,
    **kwargs: Any,
) -> Tuple[Sequence[MBConvConfig], Optional[int]]:
    inverted_residual_setting: Sequence[MBConvConfig]
    if arch.startswith("efficientnet_b"):
        bneck_conf = partial(
            MBConvConfig,
            width_mult=kwargs.pop("width_mult"),
            depth_mult=kwargs.pop("depth_mult"),
        )
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 32, 16, 1),
            bneck_conf(6, 3, 2, 16, 24, 2),
            bneck_conf(6, 5, 2, 24, 40, 2),
            bneck_conf(6, 3, 2, 40, 80, 3),
            bneck_conf(6, 5, 1, 80, 112, 3),
            bneck_conf(6, 5, 2, 112, 192, 4),
            bneck_conf(6, 3, 1, 192, 320, 1),
        ]
        last_channel = None
    else:
        raise ValueError(f"Unsupported model type {arch}")
    return inverted_residual_setting, last_channel


def filmefficientnet_b3(
    *,
    weights: Optional[EfficientNet_B3_Weights] = None,
    progress: bool = True,
    **kwargs: Any,
) -> FilmEfficientNet:
    """EfficientNet B3 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        weights (:class:`~torchvision.models.EfficientNet_B3_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.EfficientNet_B3_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.

        **kwargs: parameters passed to the ``FilmEfficientNet``
            base class.
    .. autoclass:: torchvision.models.EfficientNet_B3_Weights
        :members:
    """
    weights = EfficientNet_B3_Weights.verify(weights)

    inverted_residual_setting, last_channel = _filmefficientnet_conf(
        "efficientnet_b3", width_mult=1.2, depth_mult=1.4
    )
    return _filmefficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.3),
        last_channel,
        weights,
        progress,
        **kwargs,
    )


def decode_predictions(preds: torch.Tensor, top=5):
    preds = preds.detach().cpu().numpy()
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(_IMAGENET_CATEGORIES[i], pred[i]) for i in top_indices]
        results.append(result)
    return results
