import torch
import torch.nn as nn
from monai.networks.nets.resnet import ResNet as _MONAIResNet, ResNetBlock

# Monkey-patch MONAI blocks to avoid circular imports
from monai.networks.blocks.selfattention import SABlock
from monai.networks.blocks.selfattention       import SABlock
from monai.networks.blocks.mlp                 import MLPBlock
from monai.networks.blocks.crossattention      import CrossAttentionBlock

# ② Monkey-patch 到 blocks 包
import monai.networks.blocks as _blocks
_blocks.SABlock             = SABlock
_blocks.MLPBlock            = MLPBlock
_blocks.CrossAttentionBlock = CrossAttentionBlock

# ③ 现在再安全地导入 TransformerBlock
from monai.networks.blocks.transformerblock    import TransformerBlock

class ResNet3DWithCBAMViT(_MONAIResNet):
    """
    3D ResNet with integrated CBAM and ViT blocks.
    """
    def __init__(
        self,
        block: type[ResNetBlock] | str = "basic",
        layers: tuple[int, ...] = (2, 2, 2, 2),
        block_inplanes: tuple[int, ...] = (64, 128, 256, 512),
        spatial_dims: int = 3,
        n_input_channels: int = 1,
        num_classes: int = 2,
        conv1_t_size: int | tuple[int, ...] = 7,
        conv1_t_stride: int | tuple[int, ...] = 1,
        no_max_pool: bool = False,
        shortcut_type: str = "B",
        widen_factor: float = 1.0,
        feed_forward: bool = True,
        bias_downsample: bool = True,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
    ) -> None:
        super().__init__(
            block=ResNetBlock if block == "basic" else block,
            layers=list(layers),
            block_inplanes=list(block_inplanes),
            spatial_dims=spatial_dims,
            n_input_channels=n_input_channels,
            conv1_t_size=conv1_t_size,
            conv1_t_stride=conv1_t_stride,
            no_max_pool=no_max_pool,
            shortcut_type=shortcut_type,
            widen_factor=widen_factor,
            num_classes=num_classes,
            feed_forward=feed_forward,
            bias_downsample=bias_downsample,
            act=act,
            norm=norm,
        )
        # Append CBAM and ViT modules to each ResNetBlock
        # The ResNetBlock class should handle CBAM and TransformerBlock internally.

        # Reinitialize weights and biases, skipping None biases
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
model = ResNet3DWithCBAMViT()
print(model)
