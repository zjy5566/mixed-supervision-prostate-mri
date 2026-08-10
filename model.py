import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. Basic module: 3D residual block
# ==========================================
class ResBlock3D(nn.Module):
    """3D residual block with GroupNorm.

    GroupNorm is more stable than BatchNorm for small 3D medical-image batches.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        num_groups: int = 8,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        if out_channels % num_groups != 0:
            num_groups = 1

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=float(dropout_rate)) if dropout_rate > 0 else nn.Identity()

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.relu(out)
        return out


# ==========================================
# 2. Segmentation-only ResU-Net + MIL pooling
# ==========================================
class ProstateSegMILNet(nn.Module):
    """3D ResU-Net for lesion segmentation with zone-level MIL supervision.

    Main design:
      1. The network has only one learnable output head: voxel-level lesion logits.
      2. Region-level / systematic-biopsy predictions are obtained by pooling the
         lesion logits inside each anatomical zone.

    This makes dense radiologist annotation and weak regional biopsy supervision
    optimise the same lesion heatmap, instead of learning two unrelated heads.

    Outputs in default return_dict=True mode:
      - lesion_logits:      (B, 1, D, H, W), raw voxel logits for lesion segmentation
      - region_logits:      (B, max_zones, 1) or None, MIL-pooled zone logits
      - region_valid_mask:  (B, max_zones) or None, True for zones present in zones_mask

    Notes:
      - Apply BCEWithLogitsLoss / DiceWithLogitsLoss directly to lesion_logits.
      - Apply BCEWithLogitsLoss directly to region_logits.squeeze(-1) for regional labels.
      - Do not apply sigmoid inside the model during training.
    """

    def __init__(
        self,
        in_channels: int = 3,
        max_zones: int = 20,
        base_channels: int = 32,
        dropout_rate: float = 0.0,
        mil_pooling: str = "lme",
        lme_r: float = 8.0,
        return_dict: bool = True,
        **kwargs,
    ):
        """Create the model.

        Args:
            in_channels: number of input MRI channels, e.g. T2/DWI/ADC = 3.
            max_zones: maximum number of systematic biopsy regions, e.g. 20.
            base_channels: first-layer channel width.
            dropout_rate: 3D feature dropout probability used inside residual blocks.
            mil_pooling: 'lme', 'max', or 'mean'. 'lme' is the recommended default.
            lme_r: sharpness parameter for log-mean-exp pooling.
            return_dict: if True, return a dictionary; if False, return a compact tuple.
            **kwargs: accepted for backward compatibility, e.g. num_grade_classes is ignored.
        """
        super().__init__()
        self.max_zones = int(max_zones)
        self.mil_pooling = mil_pooling
        self.lme_r = float(lme_r)
        self.return_dict = bool(return_dict)
        self.dropout_rate = float(dropout_rate)

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        # Encoder
        self.enc1 = ResBlock3D(in_channels, c1, dropout_rate=self.dropout_rate)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = ResBlock3D(c1, c2, dropout_rate=self.dropout_rate)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = ResBlock3D(c2, c3, dropout_rate=self.dropout_rate)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc4 = ResBlock3D(c3, c4, dropout_rate=self.dropout_rate)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ResBlock3D(c4, c5, dropout_rate=self.dropout_rate)

        # Decoder
        self.up4 = nn.ConvTranspose3d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = ResBlock3D(c4 + c4, c4, dropout_rate=self.dropout_rate)

        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ResBlock3D(c3 + c3, c3, dropout_rate=self.dropout_rate)

        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ResBlock3D(c2 + c2, c2, dropout_rate=self.dropout_rate)

        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ResBlock3D(c1 + c1, c1, dropout_rate=self.dropout_rate)

        # The only learnable output head.
        self.head_lesion = nn.Conv3d(c1, 1, kernel_size=1)

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Resize x to the spatial size of ref if needed.

        This makes the U-Net robust when D/H/W are not perfectly divisible by 16.
        """
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)
        return x

    def _decode(self, e1: torch.Tensor, e2: torch.Tensor, e3: torch.Tensor, e4: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        d4 = self.up4(b)
        d4 = self._resize_like(d4, e4)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._resize_like(d3, e3)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._resize_like(d2, e2)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._resize_like(d1, e1)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)
        return d1

    def zone_mil_pooling(
        self,
        voxel_logits: torch.Tensor,
        zones_mask: torch.Tensor,
        mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool voxel lesion logits into per-zone logits.

        Args:
            voxel_logits: (B, 1, D, H, W), raw lesion logits.
            zones_mask: (B, 1, D, H, W), integer zone ids. 0 means background.
            mode: 'lme', 'max', or 'mean'. Defaults to self.mil_pooling.

        Returns:
            region_logits: (B, max_zones, 1)
            region_valid_mask: (B, max_zones), bool
        """
        mode = self.mil_pooling if mode is None else mode
        if mode not in {"lme", "max", "mean"}:
            raise ValueError(f"Unsupported MIL pooling mode: {mode}. Use 'lme', 'max', or 'mean'.")

        if zones_mask.shape[2:] != voxel_logits.shape[2:]:
            zones_mask = F.interpolate(zones_mask.float(), size=voxel_logits.shape[2:], mode="nearest")

        B, C, _, _, _ = voxel_logits.shape
        device = voxel_logits.device
        dtype = voxel_logits.dtype

        zones = zones_mask[:, 0].round().long()
        region_logits = voxel_logits.new_zeros((B, self.max_zones, C))
        region_valid_mask = torch.zeros((B, self.max_zones), dtype=torch.bool, device=device)

        for b_idx in range(B):
            logits_b = voxel_logits[b_idx]  # (1, D, H, W)
            zones_b = zones[b_idx]

            for zone_id in range(1, self.max_zones + 1):
                zone_voxels = zones_b == zone_id
                if not torch.any(zone_voxels):
                    continue

                values = logits_b[:, zone_voxels]  # (1, N)
                n_voxels = values.shape[1]

                if mode == "lme":
                    r = self.lme_r
                    pooled = torch.logsumexp(values * r, dim=1) / r - math.log(float(n_voxels)) / r
                elif mode == "max":
                    pooled = values.max(dim=1).values
                else:  # mode == "mean"
                    pooled = values.mean(dim=1)

                region_logits[b_idx, zone_id - 1] = pooled.to(dtype=dtype)
                region_valid_mask[b_idx, zone_id - 1] = True

        return region_logits, region_valid_mask

    def forward(
        self,
        x: torch.Tensor,
        zones_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Dict[str, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Bottleneck + decoder
        b = self.bottleneck(p4)
        d1 = self._decode(e1, e2, e3, e4, b)

        lesion_logits = self.head_lesion(d1)

        region_logits = None
        region_valid_mask = None
        if zones_mask is not None:
            region_logits, region_valid_mask = self.zone_mil_pooling(lesion_logits, zones_mask)

        use_dict = self.return_dict if return_dict is None else bool(return_dict)
        if not use_dict:
            return lesion_logits, region_logits, region_valid_mask

        return {
            "lesion_logits": lesion_logits,
            "lesion_pred": lesion_logits,          # backward-compatible alias
            "region_logits": region_logits,
            "sys_lesion_preds": region_logits,     # backward-compatible alias
            "region_valid_mask": region_valid_mask,
        }


# Backward-compatible class name for existing training scripts.
# Existing code can still do: from model import ProstateMixedSupervisionNet
class ProstateMixedSupervisionNet(ProstateSegMILNet):
    pass


if __name__ == "__main__":
    model = ProstateSegMILNet(in_channels=3, max_zones=20, return_dict=True)
    x = torch.randn(2, 3, 64, 96, 96)
    zones = torch.zeros(2, 1, 64, 96, 96)
    zones[:, :, :, :48, :48] = 1
    zones[:, :, :, 48:, :48] = 2
    zones[:, :, :, :48, 48:] = 3

    out = model(x, zones)
    print("lesion_logits:", tuple(out["lesion_logits"].shape))
    print("region_logits:", tuple(out["region_logits"].shape))
    print("region_valid_mask:", tuple(out["region_valid_mask"].shape))
