import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os
import datetime
import re


train_size = (1, 3, 256, 256)


class _DebugWriter:
    """Minimal debug writer that appends human-readable traces to a text file.

    Safe to call from multiple forwards; creates parent directory automatically.
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._header_written = False
        self._base_shape = None  # (B, C, H, W)
        self._tensor_sources = {}
        if os.path.exists(log_path):
            os.remove(log_path)
        parent = os.path.dirname(os.path.abspath(log_path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

    def write(self, text: str) -> None:
        mode = "a"
        with open(self.log_path, mode, encoding="utf-8") as f:
            if not self._header_written:
                f.write("\n=== SAFNet Debug Trace ===\n")
                f.write(f"Started: {datetime.datetime.now().isoformat()}\n")
                f.write("==========================\n")
                self._header_written = True
            # If base shape is known, augment any tensor shape tuples with relative notation
            line = text.rstrip("\n")
            if self._base_shape is not None:
                line = self._augment_shapes_with_relative(line)
            f.write(line + "\n")

    def set_base_shape(self, shape: tuple) -> None:
        # Expect (B, C, H, W)
        if len(shape) == 4:
            self._base_shape = tuple(int(x) for x in shape)

    def tag(self, tensor: torch.Tensor, name: str) -> None:
        try:
            self._tensor_sources[id(tensor)] = name
        except Exception:
            pass

    def source(self, tensor: torch.Tensor) -> str:
        return self._tensor_sources.get(id(tensor), None)

    def _augment_shapes_with_relative(self, line: str) -> str:
        # Replace every tuple of 4 integers in parentheses with appended relative form
        tuple_pattern = re.compile(r"\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)")

        def repl(m):
            b, c, h, w = [int(m.group(i)) for i in range(1, 5)]
            rel = self._format_relative((b, c, h, w))
            return f"({b}, {c}, {h}, {w}) => {rel}"

        return tuple_pattern.sub(repl, line)

    def _format_relative(self, shape: tuple) -> str:
        # Relative to input: show as (B, C, W_rel, H_rel) with W/H derived from input W0/H0
        if self._base_shape is None or len(shape) != 4:
            return str(shape)
        b0, c0, h0, w0 = self._base_shape
        b, c, h, w = shape

        def rel_wh(cur, base, symbol):
            if base <= 0:
                return f"{cur}"
            if cur == base:
                return symbol
            if base % cur == 0:
                return f"{symbol}/{base // cur}"
            if cur % base == 0:
                return f"{symbol}*{cur // base}"
            # Fallback decimal ratio
            ratio = cur / base
            return f"{symbol}*{ratio:.2f}"

        # Channels as absolute numeric value
        c_str = str(c)
        # Note: order as (W, H) per user request
        w_rel = rel_wh(w, w0, "W")
        h_rel = rel_wh(h, h0, "H")
        return f"(B, {c_str}, {w_rel}, {h_rel})"


def _attach_debug(module: nn.Module, writer: _DebugWriter, name: str = None) -> None:
    """Attach debug flags and writer to a module instance dynamically."""
    setattr(module, "debug_enabled", True)
    setattr(module, "_debug", writer)
    if name is not None:
        setattr(module, "debug_name", name)


def _log(module: nn.Module, message: str, add_spacing: bool = False) -> None:
    """Write a debug message if module has debugging enabled."""
    if getattr(module, "debug_enabled", False) and getattr(module, "_debug", None):
        prefix = getattr(module, "debug_name", module.__class__.__name__)
        module._debug.write(f"[{prefix}] {message}")
        if add_spacing:
            module._debug.write("")  # Add blank line for spacing


class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]

    def extra_repr(self) -> str:
        return "kernel_size={}, base_size={}, stride={}, fast_imp={}".format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = (
                    min(h - 1, self.kernel_size[0] // r1),
                    min(w - 1, self.kernel_size[1] // r2),
                )
                out = (
                    s[:, :, :-k1, :-k2]
                    - s[:, :, :-k1, k2:]
                    - s[:, :, k1:, :-k2]
                    + s[:, :, k1:, k2:]
                ) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = (
                s[:, :, :-k1, :-k2],
                s[:, :, :-k1, k2:],
                s[:, :, k1:, :-k2],
                s[:, :, k1:, k2:],
            )
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode="replicate")

        return out


class BasicConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride,
        bias=True,
        norm=False,
        relu=True,
        transpose=False,
    ):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                )
            )
        else:
            if stride == 2:
                layers.append(
                    nn.Conv2d(
                        in_channel,
                        out_channel,
                        kernel_size,
                        padding=padding,
                        stride=stride,
                        bias=bias,
                    )
                )
            else:
                nn.Conv2d(
                    in_channel,
                    in_channel,
                    kernel_size,
                    groups=in_channel,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                )
                layers.append(
                    nn.Conv2d(
                        in_channel,
                        out_channel,
                        1,
                        padding=0,
                        stride=1,
                        bias=bias,
                    )
                )
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        _log(self, f"in: {tuple(x.shape)}")
        out = self.main(x)
        _log(self, f"out: {tuple(out.shape)}", add_spacing=True)
        return out


class Gap(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()

        self.fscale_d = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        self.fscale_h = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        _log(self, f"in: {tuple(x.shape)}")
        x_d = self.gap(x)
        _log(self, f"gap: {tuple(x_d.shape)}")
        x_h = (x - x_d) * (self.fscale_h[None, :, None, None] + 1.0)
        _log(self, f"high_freq: {tuple(x_h.shape)}")
        x_d = x_d * self.fscale_d[None, :, None, None]
        out = x_d + x_h
        _log(self, f"out: {tuple(out.shape)}", add_spacing=True)
        return out


class Patch_ap(nn.Module):
    def __init__(self, inchannel, patch_size):
        super(Patch_ap, self).__init__()

        self.ap = nn.AdaptiveAvgPool2d((1, 1))

        self.patch_size = patch_size
        self.channel = inchannel * patch_size**2
        self.h = nn.Parameter(torch.zeros(self.channel))
        self.l = nn.Parameter(torch.zeros(self.channel))

    def forward(self, x):
        _log(self, f"in: {tuple(x.shape)}")
        patch_x = rearrange(
            x,
            "b c (p1 w1) (p2 w2) -> b c p1 w1 p2 w2",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        _log(self, f"patch_reshape1: {tuple(patch_x.shape)}")
        patch_x = rearrange(
            patch_x,
            " b c p1 w1 p2 w2 -> b (c p1 p2) w1 w2",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        _log(self, f"patch_reshape2: {tuple(patch_x.shape)}")

        low = self.ap(patch_x)
        _log(self, f"low_freq: {tuple(low.shape)}")
        high = (patch_x - low) * self.h[None, :, None, None]
        _log(self, f"high_freq: {tuple(high.shape)}")
        out = high + low * self.l[None, :, None, None]
        out = rearrange(
            out,
            "b (c p1 p2) w1 w2 -> b c (p1 w1) (p2 w2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        _log(self, f"out: {tuple(out.shape)}", add_spacing=True)
        return out


class SpatialChannelModulator(nn.Module):
    """
    Replacement for SFconv: combines channel-wise attention with spatial gating.
    Input: low (B,C,H,W), high (B,C,H,W)
    Output: fused feature (B,C,H,W)
    """

    def __init__(self, features, M=2, r=2, L=32):
        super().__init__()
        d = max(int(features // r), L)
        self.features = features
        # channel pathway (like SFconv)
        self.fc = nn.Conv2d(features, d, 1, 1, 0)
        self.fcs = nn.ModuleList([nn.Conv2d(d, features, 1, 1, 0) for _ in range(M)])
        self.softmax = nn.Softmax(dim=1)
        self.out = nn.Conv2d(features, features, 1, 1, 0)
        # spatial pathway
        # small conv stack producing 1-channel spatial map
        self.spatial = nn.Sequential(
            nn.Conv2d(
                features,
                features // 2 if features // 2 > 0 else 1,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(features // 2 if features // 2 > 0 else 1)
            if features // 2 > 0
            else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(
                features // 2 if features // 2 > 0 else 1, 1, kernel_size=1, bias=True
            ),
            nn.Sigmoid(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, low, high):
        _log(self, f"in low: {tuple(low.shape)}, high: {tuple(high.shape)}")
        # low, high: (B,C,H,W)
        fused = low + high  # (B,C,H,W)
        _log(self, f"fused: {tuple(fused.shape)}")
        # channel path
        z = self.gap(fused)  # (B,C,1,1)
        _log(self, f"gap: {tuple(z.shape)}")
        fea_z = self.fc(z)  # (B,d,1,1)
        _log(self, f"fc: {tuple(fea_z.shape)}")
        high_att = self.fcs[0](fea_z)  # (B,C,1,1)
        low_att = self.fcs[1](fea_z)  # (B,C,1,1)
        _log(
            self,
            f"attentions: high={tuple(high_att.shape)}, low={tuple(low_att.shape)}",
        )
        attention_vectors = torch.cat([high_att, low_att], dim=1)  # (B,2C,1,1)
        attention_vectors = self.softmax(attention_vectors)  # softmax across 2C
        high_att, low_att = torch.chunk(attention_vectors, 2, dim=1)  # each (B,C,1,1)
        _log(
            self,
            f"softmax_attentions: high={tuple(high_att.shape)}, low={tuple(low_att.shape)}",
        )

        # expand to spatial
        high_att_spatial = high_att.expand_as(fused)  # (B,C,H,W)
        low_att_spatial = low_att.expand_as(fused)
        _log(
            self,
            f"spatial_attentions: high={tuple(high_att_spatial.shape)}, low={tuple(low_att_spatial.shape)}",
        )

        # spatial path
        spatial_map = self.spatial(fused)  # (B,1,H,W) in (0,1)
        _log(self, f"spatial_map: {tuple(spatial_map.shape)}")
        # apply spatial gating: modulate channel weights by spatial map
        high_att_spatial = high_att_spatial * spatial_map
        low_att_spatial = low_att_spatial * (1 - spatial_map)  # complementary
        _log(
            self,
            f"gated_attentions: high={tuple(high_att_spatial.shape)}, low={tuple(low_att_spatial.shape)}",
        )

        # apply
        fea_high = high * high_att_spatial
        fea_low = low * low_att_spatial
        _log(
            self,
            f"gated_features: high={tuple(fea_high.shape)}, low={tuple(fea_low.shape)}",
        )

        out = self.out(fea_high + fea_low)
        _log(self, f"out: {tuple(out.shape)}", add_spacing=True)
        return out


class sMCSF(nn.Module):
    """
    small Multi-scale Contextual Feature module: pyramid pooling + lightweight windowed interaction
    Returns context feature same spatial size as input.
    """

    def __init__(self, in_ch, out_ch, pool_sizes=(1, 2, 4)):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((ps, ps)),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                    nn.GELU(),
                )
                for ps in pool_sizes
            ]
        )
        # ASPP-lite
        self.aspp = nn.ModuleList(
            [
                nn.Conv2d(
                    in_ch, out_ch, kernel_size=3, padding=r, dilation=r, bias=False
                )
                for r in (1, 6, 12)
            ]
        )
        self.merge = nn.Sequential(
            nn.Conv2d(
                out_ch * (len(pool_sizes) + len(self.aspp)),
                out_ch,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        pooled = []
        for seq in self.convs:
            p = seq[0](x)  # adaptive pool -> size ps x ps
            p = seq[1](p)  # conv1x1
            p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)
            pooled.append(p)
        aspp_feats = [
            F.interpolate(conv(x), size=(h, w), mode="bilinear", align_corners=False)
            for conv in self.aspp
        ]
        feats = pooled + aspp_feats
        out = torch.cat(feats, dim=1)
        out = self.merge(out)
        return out


class dynamic_filter(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)

        # filter generator from global statistics (kept simple & efficient)
        self.conv = nn.Conv2d(
            inchannels, group * kernel_size**2, kernel_size=1, stride=1, bias=False
        )
        # self.bn = nn.InstanceNorm2d(group*kernel_size**2)
        # softmax across kernel positions
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

        self.pad = nn.ReflectionPad2d(kernel_size // 2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        # use SpatialChannelModulator instead of original SFconv
        self.modulate = SpatialChannelModulator(inchannels)

    def forward(self, x):
        identity_input = x
        # generate low-pass filter from global pooled descriptor
        low_filter = self.ap(x)  # (B, C_g, 1, 1)
        low_filter = self.conv(low_filter)  # (B, group*k^2, 1, 1)
        # low_filter = self.bn(low_filter)              # BN

        n, c, h, w = x.shape
        # extract local patches
        x_unf = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(
            n, self.group, c // self.group, self.kernel_size**2, h * w
        )
        # reshape low_filter to multiply: (B, groups, 1, k^2, 1)
        n, c1, p, q = low_filter.shape
        # low_filter currently (B, group*k^2, 1, 1)
        low_filter = low_filter.reshape(
            n, self.group, self.kernel_size**2, 1
        ).unsqueeze(2)  # (B,group,1,k^2,1)
        low_filter = self.act(low_filter)  # softmax over k^2
        # broadcast multiply & sum -> low_part
        # x_unf: (B, group, Cg_per_group, k^2, h*w)
        low_part = torch.sum(x_unf * low_filter, dim=3).reshape(n, c, h, w)

        # high part
        out_high = identity_input - low_part
        out = self.modulate(low_part, out_high)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(
            in_channel, out_channel, kernel_size=3, stride=1, relu=True
        )
        self.conv2 = BasicConv(
            out_channel, out_channel, kernel_size=3, stride=1, relu=False
        )
        self.filter = filter

        self.dyna = dynamic_filter(in_channel // 2) if filter else nn.Identity()
        self.dyna_2 = (
            dynamic_filter(in_channel // 2, kernel_size=5) if filter else nn.Identity()
        )

        self.localap = Patch_ap(in_channel // 2, patch_size=2)
        self.global_ap = Gap(in_channel // 2)

    def forward(self, x):
        _log(self, f"in: {tuple(x.shape)}")
        out = self.conv1(x)
        _log(self, f"conv1: {tuple(out.shape)}")

        if self.filter:
            k3, k5 = torch.chunk(out, 2, dim=1)
            _log(self, f"chunk: k3={tuple(k3.shape)}, k5={tuple(k5.shape)}")
            out_k3 = self.dyna(k3)
            out_k5 = self.dyna_2(k5)
            _log(
                self,
                f"dynamic_filters: k3={tuple(out_k3.shape)}, k5={tuple(out_k5.shape)}",
            )
            out = torch.cat((out_k3, out_k5), dim=1)
            _log(self, f"filtered: {tuple(out.shape)}")

        non_local, local = torch.chunk(out, 2, dim=1)
        _log(
            self,
            f"split: non_local={tuple(non_local.shape)}, local={tuple(local.shape)}",
        )
        non_local = self.global_ap(non_local)
        local = self.localap(local)
        _log(
            self,
            f"processed: non_local={tuple(non_local.shape)}, local={tuple(local.shape)}",
        )
        out = torch.cat((non_local, local), dim=1)
        _log(self, f"concat: {tuple(out.shape)}")
        out = self.conv2(out)
        _log(self, f"conv2: {tuple(out.shape)}")
        result = out + x
        _log(self, f"residual: {tuple(result.shape)}", add_spacing=True)
        return result


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res - 1)]
        layers.append(ResBlock(out_channel, out_channel, filter=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        _log(self, f"in: {tuple(x.shape)}")
        out = self.layers(x)
        _log(self, f"out: {tuple(out.shape)}", add_spacing=True)
        return out


class DBlock(nn.Module):
    def __init__(self, channel, num_res):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res - 1)]
        layers.append(ResBlock(channel, channel, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        _log(self, f"in: {tuple(x.shape)}")
        out = self.layers(x)
        _log(self, f"out: {tuple(out.shape)}", add_spacing=True)
        return out


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        # x (B, C, H, W)
        self.main = nn.Sequential(
            BasicConv(
                3, out_plane // 4, kernel_size=3, stride=1, relu=True
            ),  # (B, C, H, W) => (B, C/4, H, W)
            BasicConv(
                out_plane // 4,
                out_plane // 2,
                kernel_size=3,
                stride=1,
                relu=True,  # (B, C/4, H, W) => (B, C/2, H, W)
            ),
            # BasicConv(
            #     out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True
            # ),
            BasicConv(
                out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False
            ),  # (B, C/2, H, W) => (B, C, H, W)
            nn.InstanceNorm2d(out_plane, affine=True),
        )

    def forward(self, x):
        _log(self, f"in: {tuple(x.shape)}")
        out = self.main(x)
        _log(self, f"out: {tuple(out.shape)}", add_spacing=True)
        return out


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(
            channel * 2, channel, kernel_size=1, stride=1, relu=False
        )

    def forward(self, x1, x2):
        _log(self, f"in: x1={tuple(x1.shape)}, x2={tuple(x2.shape)}")
        concat = torch.cat([x1, x2], dim=1)
        _log(self, f"concat: {tuple(concat.shape)}")
        out = self.merge(concat)
        _log(self, f"out: {tuple(out.shape)}", add_spacing=True)
        return out


class OutPut(nn.Module):
    def __init__(self, in_chs):
        super(OutPut, self).__init__()
        self.out = nn.Sequential(
            nn.Conv2d(in_chs, in_chs, 1, bias=False),
            nn.BatchNorm2d(in_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chs, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat):
        _log(self, f"in: {tuple(feat.shape)}")
        out = self.out(feat)
        _log(self, f"out: {tuple(out.shape)}", add_spacing=True)
        return out


class SAFNet(nn.Module):
    def __init__(
        self,
        num_res=8,
        base_channel=32,
        embed_dim=128,
        debug_enabled=False,
        debug_log_path="./safnet_debug.txt",
        debug_attach_all=True,
        **kwargs,
    ):
        """
        mode: tuple like ('train', 'HIDE') or ('test','HIDE') as in SFNet codebase
        num_res: number of resblocks per level (kept smaller by default)
        base_channel: base number of filters
        embed_dim: dimension of frequency embedding d
        debug_enabled: enable debug logging
        debug_log_path: path for debug log file
        debug_attach_all: attach debug to all modules
        """
        super(SAFNet, self).__init__()
        self.embed_dim = embed_dim

        # Debug configuration
        self.debug_enabled = debug_enabled
        self._debug = _DebugWriter(debug_log_path) if debug_enabled else None

        self.Encoder = nn.ModuleList(
            [
                EBlock(base_channel, num_res),
                EBlock(base_channel * 2, num_res),
                EBlock(base_channel * 4, num_res),
            ]
        )

        # Thay thế BasicConv bằng Conv2d đơn giản
        self.feat_extract = nn.ModuleList(
            [
                # Layer 0: 3→32, stride=1 (không đổi)
                nn.Conv2d(3, base_channel, 3, stride=1, padding=1),
                # Layer 1: 32→64, stride=2 (downsample)
                nn.Conv2d(base_channel, base_channel * 2, 3, stride=2, padding=1),
                # Layer 2: 64→128, stride=2 (downsample)
                nn.Conv2d(base_channel * 2, base_channel * 4, 3, stride=2, padding=1),
                # Layer 3: 128→64, stride=2 transpose (upsample)
                nn.ConvTranspose2d(
                    base_channel * 4, base_channel * 2, 4, stride=2, padding=1
                ),
                # Layer 4: 64→32, stride=2 transpose (upsample)
                nn.ConvTranspose2d(
                    base_channel * 2, base_channel, 4, stride=2, padding=1
                ),
                # Layer 5: 32→1 (final output)
                nn.Conv2d(base_channel, 1, 3, stride=1, padding=1),
            ]
        )

        self.Decoder = nn.ModuleList(
            [
                DBlock(base_channel * 4, num_res),
                DBlock(base_channel * 2, num_res),
                DBlock(base_channel, num_res),
            ]
        )

        self.Convs = nn.ModuleList(
            [
                BasicConv(
                    base_channel * 4,
                    base_channel * 2,
                    kernel_size=1,
                    relu=True,
                    stride=1,
                ),
                BasicConv(
                    base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1
                ),
            ]
        )

        # outputs now 1 channel each (mask logits)
        self.ConvsOut = nn.ModuleList(
            [
                OutPut(base_channel * 4),
                OutPut(base_channel * 2),
            ]
        )

        # original SCM + FAM for multi-input merging (kept)
        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        # new modules:
        # small multi-scale context module
        self.sMCSF1 = sMCSF(base_channel * 4, base_channel * 4)
        self.sMCSF2 = sMCSF(base_channel * 2, base_channel * 2)

        # projection for high-frequency + context merge (registered to move with model)
        self.hf_proj = BasicConv(
            base_channel * 8,  # concat of res3 (C=4B) and ctx_deep (C=4B)
            base_channel * 4,
            kernel_size=1,
            stride=1,
            relu=True,
        )

        # Attach debug to all modules if enabled
        if self.debug_enabled and self._debug is not None and debug_attach_all:
            self._attach_debug_to_all()

    def forward(self, x):
        if self.debug_enabled:
            self._debug.write("\n-- SAFNet Forward pass --")
            self._debug.set_base_shape(tuple(x.shape))
            self._debug.write(
                f"input: {tuple(x.shape)} (min={x.min().item():.4f}, max={x.max().item():.4f})"
            )

        # multi-inputs
        x_2 = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        x_4 = F.interpolate(x_2, scale_factor=0.5, mode="bilinear", align_corners=False)
        if self.debug_enabled:
            self._debug.write(
                f"multi_inputs: x_2={tuple(x_2.shape)}, x_4={tuple(x_4.shape)}"
            )

        # image-level SCM (legacy) for multi-input fusion
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)
        if self.debug_enabled:
            self._debug.write(
                f"SCM outputs: z2={tuple(z2.shape)}, z4={tuple(z4.shape)}"
            )

        masks = []
        # Encode
        x_ = self.feat_extract[0](x)  # full res conv -> base_channel
        if self.debug_enabled:
            self._debug.write(f"feat_extract[0]: {tuple(x_.shape)}")
        res1 = self.Encoder[0](x_)  # output at full res (C)
        if self.debug_enabled:
            self._debug.write(f"Encoder[0]: {tuple(res1.shape)}")
        z = self.feat_extract[1](res1)  # downsample by 2
        if self.debug_enabled:
            self._debug.write(f"feat_extract[1]: {tuple(z.shape)}")
        z = self.FAM2(z, z2)
        if self.debug_enabled:
            self._debug.write(f"FAM2: {tuple(z.shape)}")
        res2 = self.Encoder[1](z)  # mid res
        if self.debug_enabled:
            self._debug.write(f"Encoder[1]: {tuple(res2.shape)}")
        z = self.feat_extract[2](res2)  # downsample by 2
        if self.debug_enabled:
            self._debug.write(f"feat_extract[2]: {tuple(z.shape)}")
        z = self.FAM1(z, z4)
        if self.debug_enabled:
            self._debug.write(f"FAM1: {tuple(z.shape)}")
        res3 = self.Encoder[2](z)  # deepest (coarse)
        if self.debug_enabled:
            self._debug.write(f"Encoder[2]: {tuple(res3.shape)}")

        # apply sMCSF context on deepest and mid features (helps SCM decisions)
        ctx_deep = self.sMCSF1(res3)
        ctx_mid = self.sMCSF2(res2)
        if self.debug_enabled:
            self._debug.write(
                f"sMCSF: ctx_deep={tuple(ctx_deep.shape)}, ctx_mid={tuple(ctx_mid.shape)}"
            )

        # For FEH/EdgeHead we compute a simple high-frequency approximation:
        # high_freq = feature - local_avg(feature)
        # use deepest feature res3 aggregated with ctx_deep
        hf_agg = res3 - F.avg_pool2d(res3, kernel_size=3, padding=1, stride=1)
        # pass through a small conv to merge with ctx_deep before heads
        hf_merge = torch.cat([hf_agg, ctx_deep], dim=1)
        # adapt back to expected channels (base_channel*4)
        hf_feat = self.hf_proj(hf_merge)
        # Decode
        z = self.Decoder[0](hf_feat)  # decode deepest
        z_ = self.ConvsOut[0](z)  # logits at mid resolution (1 channel)
        z_up = self.feat_extract[3](z)  # upsample
        # note: we add skip from x_4 (image-level) as in original SFNet
        masks.append(z_)

        z = torch.cat([z_up, ctx_mid], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_2 = self.ConvsOut[1](z)
        z_up2 = self.feat_extract[4](z)
        masks.append(z_2)

        z = torch.cat([z_up2, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        masks.append(z)  # final logits full res

        if self.debug_enabled:
            self._debug.write(
                f"final outputs: masks={len(masks)}, pred_masks={tuple(z.shape)}"
            )

        return {
            "masks": masks,  # list of 3 logits (coarse->fine)
            # "edge": edge_logits,  # coarse edge logits (corresponding to res3 spatial size)
            # "freq_embed": freq_embed,
            "pred_masks": z,
        }

    def _attach_debug_to_all(self) -> None:
        """Attach debug functionality to all modules with 4-level depth hierarchy."""
        writer = self._debug

        # Level 1: Main network components
        _attach_debug(self, writer, name="SAFNet")

        # Level 2: Encoders and Decoders
        for i, enc in enumerate(self.Encoder):
            _attach_debug(enc, writer, name=f"Encoder[{i}]")
        for i, dec in enumerate(self.Decoder):
            _attach_debug(dec, writer, name=f"Decoder[{i}]")

        # Level 3: Feature extraction and processing
        for i, feat in enumerate(self.feat_extract):
            _attach_debug(feat, writer, name=f"FeatExtract[{i}]")
        for i, conv in enumerate(self.Convs):
            _attach_debug(conv, writer, name=f"Convs[{i}]")
        for i, conv_out in enumerate(self.ConvsOut):
            _attach_debug(conv_out, writer, name=f"ConvsOut[{i}]")

        # Level 3: SCM and FAM modules
        _attach_debug(self.SCM1, writer, name="SCM1")
        _attach_debug(self.SCM2, writer, name="SCM2")
        _attach_debug(self.FAM1, writer, name="FAM1")
        _attach_debug(self.FAM2, writer, name="FAM2")

        # Level 3: Context modules
        _attach_debug(self.sMCSF1, writer, name="sMCSF1")
        _attach_debug(self.sMCSF2, writer, name="sMCSF2")
        _attach_debug(self.hf_proj, writer, name="HFProj")

        # Level 4: Individual ResBlocks within Encoders/Decoders
        for i, enc in enumerate(self.Encoder):
            for j, res_block in enumerate(enc.layers):
                _attach_debug(res_block, writer, name=f"Enc[{i}].ResBlock[{j}]")
                # Level 4: Components within ResBlock
                _attach_debug(
                    res_block.conv1, writer, name=f"Enc[{i}].ResBlock[{j}].Conv1"
                )
                _attach_debug(
                    res_block.conv2, writer, name=f"Enc[{i}].ResBlock[{j}].Conv2"
                )
                if hasattr(res_block, "dyna") and res_block.dyna is not None:
                    _attach_debug(
                        res_block.dyna, writer, name=f"Enc[{i}].ResBlock[{j}].Dyna"
                    )
                if hasattr(res_block, "dyna_2") and res_block.dyna_2 is not None:
                    _attach_debug(
                        res_block.dyna_2, writer, name=f"Enc[{i}].ResBlock[{j}].Dyna2"
                    )
                _attach_debug(
                    res_block.localap, writer, name=f"Enc[{i}].ResBlock[{j}].LocalAP"
                )
                _attach_debug(
                    res_block.global_ap, writer, name=f"Enc[{i}].ResBlock[{j}].GlobalAP"
                )

        for i, dec in enumerate(self.Decoder):
            for j, res_block in enumerate(dec.layers):
                _attach_debug(res_block, writer, name=f"Dec[{i}].ResBlock[{j}]")
                # Level 4: Components within ResBlock
                _attach_debug(
                    res_block.conv1, writer, name=f"Dec[{i}].ResBlock[{j}].Conv1"
                )
                _attach_debug(
                    res_block.conv2, writer, name=f"Dec[{i}].ResBlock[{j}].Conv2"
                )
                if hasattr(res_block, "dyna") and res_block.dyna is not None:
                    _attach_debug(
                        res_block.dyna, writer, name=f"Dec[{i}].ResBlock[{j}].Dyna"
                    )
                if hasattr(res_block, "dyna_2") and res_block.dyna_2 is not None:
                    _attach_debug(
                        res_block.dyna_2, writer, name=f"Dec[{i}].ResBlock[{j}].Dyna2"
                    )
                _attach_debug(
                    res_block.localap, writer, name=f"Dec[{i}].ResBlock[{j}].LocalAP"
                )
                _attach_debug(
                    res_block.global_ap, writer, name=f"Dec[{i}].ResBlock[{j}].GlobalAP"
                )


if __name__ == "__main__":
    model = SAFNet(
        num_res=4,
        base_channel=64,
        embed_dim=128,
        debug_enabled=False,
        debug_log_path="./safnet_debug.txt",
    )
    x = torch.randn(1, 3, 256, 256)
    gt_mask = torch.randn(1, 1, 256, 256)
    y = model(x)
    print(
        y["masks"][0].shape,
        y["masks"][1].shape,
        y["masks"][2].shape,
    )

    from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table

    flops = FlopCountAnalysis(model, x)
    print(flop_count_table(flops, max_depth=5, show_param_shapes=True))
    # print(parameter_count_table(model, max_depth=4))

    # conv = BasicConv(3, 64, kernel_size=3, stride=2, relu=True)
    # conv = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
    # x = torch.randn(1, 3, 512, 512)
    # print(conv(x).shape)
