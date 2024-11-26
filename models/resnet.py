# Adapted from https://github.com/lucidrains/make-a-video-pytorch/blob/main/make_a_video_pytorch/make_a_video.py
# Borrowed from https://github.com/xuduo35/MakeLongVideo/blob/main/makelongvideo/models/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConv(nn.Module):
    def __init__(self, dim, dim_out, kernel_size, temporal_num_layers, temporal_kernel_size, **kwargs):
        super().__init__()

        self.spatial_conv = nn.Conv2d(dim, dim_out, kernel_size, **kwargs)
        self.temporal_convs = Conv3dBlocks(dim_out, num_layers=temporal_num_layers, kernel_size=temporal_kernel_size, dropout=0.1)

    def forward(self, hidden_states):
        batch, channel, frames, height, width = hidden_states.shape
        
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
        hidden_states = hidden_states.reshape(batch * frames, channel, height, width)
        hidden_states = self.spatial_conv(hidden_states)
        hidden_states = hidden_states.view(batch, frames, hidden_states.shape[1], hidden_states.shape[2], hidden_states.shape[3]).contiguous()
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
        hidden_states = self.temporal_convs(hidden_states)

        return hidden_states

class Conv3dBlocks(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int = 1,
        kernel_size: int = 3,
        dropout: float = 0.1,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.GroupNorm(min(norm_num_groups, dim), dim),
                nn.SiLU(),
                nn.Dropout(dropout) if i > 0 else nn.Identity(), 
                nn.Conv3d(dim, dim, (self.kernel_size, 1, 1), padding=(self.padding, 0, 0))
            )
            self.convs.append(layer)

        nn.init.zeros_(self.convs[-1][-1].weight)
        nn.init.zeros_(self.convs[-1][-1].bias)

    def forward(self, hidden_states):
        residual = hidden_states

        for conv in self.convs:
            hidden_states = conv(hidden_states)

        hidden_states = residual + hidden_states
        return hidden_states

class Upsample(nn.Module):
    def __init__(self, 
                 channels, 
                 use_conv=False, 
                 use_conv_transpose=False, 
                 out_channels=None, 
                 name="conv"
                 ):
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        conv = None
        if use_conv_transpose:
            raise NotImplementedError
        elif use_conv:
            conv = TemporalConv(self.channels, self.out_channels, 3, temporal_num_layers=2, temporal_kernel_size=5, padding=1)

        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, 
                hidden_states, 
                output_size=None
                ):
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            raise NotImplementedError

        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=[1.0, 2.0, 2.0], mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states

class Downsample(nn.Module):
    def __init__(self, 
                 channels, 
                 use_conv=False, 
                 out_channels=None, 
                 padding=1, 
                 name="conv"
                 ):
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            conv = TemporalConv(self.channels, self.out_channels, 3, temporal_num_layers=2, temporal_kernel_size=5, stride=stride, padding=padding)
        else:
            raise NotImplementedError

        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, 
                hidden_states
                ):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            raise NotImplementedError

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)

        return hidden_states

class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        output_scale_factor: float = 1.0,
        use_in_shortcut=None,
    ):
        super().__init__()
        
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = TemporalConv(in_channels, out_channels, kernel_size=3, temporal_num_layers=1, temporal_kernel_size=7, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

            self.time_emb_proj = torch.nn.Linear(temb_channels, time_emb_proj_out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = TemporalConv(out_channels, out_channels, kernel_size=3, temporal_num_layers=4, temporal_kernel_size=3, stride=1, padding=1)

        self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = TemporalConv(in_channels, out_channels, kernel_size=1, temporal_num_layers=4, temporal_kernel_size=3, stride=1, padding=0)

    def forward(self, 
                input_tensor, 
                temb
                ):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor