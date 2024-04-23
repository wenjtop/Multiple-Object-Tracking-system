import math
import torch
import torch.nn as nn
from thop import profile
from thop import clever_format
from timm.models.layers import trunc_normal_, DropPath
import time

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()

        self.Embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.Embed(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    h_window_num = H // window_size[0]
    w_window_num = W // window_size[1]
    x = x.view(B, h_window_num, window_size[0], w_window_num, window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, h_window_num * w_window_num, window_size[0] * window_size[1], C)  # B, hn*wn, w*w, C
    return x

def window_reverse(x, window_size, H, W):
    B, N, S, C = x.shape
    h_window_num = H // window_size[0]
    w_window_num = W // window_size[1]
    x = x.view(B, h_window_num, w_window_num, window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, h_window_num*window_size[0], w_window_num*window_size[1], C)  # B, H, W, C
    return x

class PatchMerging(nn.Module):

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # self.reduction = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)

        return x

class Attention(nn.Module):
    def __init__(self, dim, window_size=7, num_heads=8, qkv_bias=True, drop_path=0., act=nn.GELU):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.scale = (dim // num_heads) ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            act(),
            nn.Linear(4 * dim, dim),
        )
        # table of relative position
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)                                  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()            # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1                           # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)                          # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)   # 训练是不更新
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def generate_mask(self, h, w, w_h, w_w):

        attn_mask = torch.zeros(h, w, w_h, w_w, w_h, w_w, dtype=torch.bool, device=self.relative_position_index.device)


        s1 = w_h - w_h//2
        s2 = w_w - w_w//2
        attn_mask[-1, :, :s1, :, s1:, :] = True
        attn_mask[-1, :, s1:, :, :s1, :] = True
        attn_mask[:, -1, :, :s2, :, s2:] = True
        attn_mask[:, -1, :, s2:, :, :s2] = True

        attn_mask = attn_mask.view(1, 1, h*w, w_h*w_w, w_h*w_w)


        return attn_mask

    def forward(self, x, reshape, type=None):
        B, N, S, C = x.shape           # B, N, S, C
        shortcut = x
        x = self.norm1(x)
        qkv = self.qkv(x).reshape(B, N, S, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(1)

        if type == 'S':
            attn_mask = self.generate_mask(int(reshape[1]/self.window_size[0]), int(reshape[2]/self.window_size[1]), self.window_size[0], self.window_size[1])
            attn = attn.masked_fill_(attn_mask, float("-inf"))

        attn = self.softmax(attn)
        x = (attn @ v).permute(0, 2, 3, 1, 4).contiguous()
        x = x.view(B, N, S, C)
        x = self.proj(x)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class LNGAttentionLayer(nn.Module):
    def __init__(self, dim, window_size, stage, num_heads, drop_path):
        super().__init__()
        self.window_size = window_size
        self.shift_size = [window_size[0]//2, window_size[0]//2]
        self.window_stage_size = 8//(2**stage)
        self.window_reverse = window_reverse
        self.window_partition = window_partition
        self.LocalAttention = Attention(dim=dim, window_size=window_size, num_heads=num_heads, drop_path=drop_path[0])
        self.NeighborAttention = Attention(dim=dim, window_size=window_size, num_heads=num_heads, drop_path=drop_path[1])
        self.GlobalAttention = Attention(dim=dim, window_size=window_size, num_heads=num_heads, drop_path=drop_path[2])

    def forward(self, x):
        B, H, W, C = x.shape
        # Local
        x = self.window_partition(x, self.window_size)            # input：B, H, W ,C。 return：B, N, S, C
        x = self.LocalAttention(x, [B, H, W, C])
        x = self.window_reverse(x, self.window_size, H, W)        # input：B, N, S, C  。return：B, C, H, W

        # Neighbor
        x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[0]), dims=(1, 2))
        x = self.window_partition(x, self.window_size)            # input：B, H, W ,C。 return：B, N, S, C
        x = self.NeighborAttention(x, [B, H, W, C], type='S')
        x = self.window_reverse(x, self.window_size, H, W)        # input：B, N, S, C  。return：B, C, H, W
        x = torch.roll(x, shifts=(self.shift_size[0], self.shift_size[0]), dims=(1, 2))

        # Global

        x = self.window_partition(x, [H//self.window_size[0], W//self.window_size[1]])      # input：B, H, W ,C。 return：B, N, S, C
        x = x.permute(0, 2, 1, 3)
        x = self.GlobalAttention(x, [B, H, W, C])
        x = x.permute(0, 2, 1, 3)
        x = self.window_reverse(x, [H//self.window_size[0], W//self.window_size[1]], H, W)  # input：B, N, S, C  。return：B, C, H, W

        return x

class LLAttentionLayer(nn.Module):
    def __init__(self, dim, window_size, stage, num_heads, drop_path):
        super().__init__()
        self.window_size = window_size
        self.window_stage_size = 8//(2**stage)
        self.window_partition = window_partition
        self.window_reverse = window_reverse
        self.LocalAttention1 = Attention(dim=dim, window_size=window_size, num_heads=num_heads, drop_path=drop_path[0])
        self.LocalAttention2 = Attention(dim=dim, window_size=window_size, num_heads=num_heads, drop_path=drop_path[1])


    def forward(self, x):
        B, H, W, C = x.shape

        x = self.window_partition(x, self.window_size)      # input：B, H, W ,C。 return：B, N, S, C
        x = self.LocalAttention1(x, [B, H, W, C])
        x = self.LocalAttention2(x, [B, H, W, C])
        x = self.window_reverse(x, self.window_size, H, W)  # input：B, N, S, C  。return：B, C, H, W

        return x

class LNGTransformer(nn.Module):
    def __init__(self, in_chans, dims, patch_size, window_size, stages, num_heads, drop_rate=0.2, num_classes=1000):
        super().__init__()
        self.dims = dims
        self.Embed = nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size)
        self.norm1 = nn.LayerNorm(dims[0])
        drop_path = [x.item() for x in torch.linspace(0, drop_rate, sum(stages)*3)]

        self.AttentionStages = nn.ModuleList()
        for i, stage in enumerate(stages[:-1]):
            for _ in range(stage):
                self.AttentionStages.append(LNGAttentionLayer(dim=dims[i], window_size=window_size, stage=i, num_heads=num_heads[i], drop_path=drop_path[i:i+3]))
            if i != len(stages)-1:
                self.AttentionStages.append(PatchMerging(dim=dims[i]))
        self.AttentionStages.append(LLAttentionLayer(dim=dims[-1], window_size=window_size, stage=len(stages)-1, num_heads=num_heads[-1], drop_path=drop_path[-3:]))
        self.norm2 = nn.LayerNorm(dims[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):                    # B, C, H, W
        x = self.Embed(x)                    # B, 96, H/4, W/4
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        B, H, W, C = x.shape

        for blk in self.AttentionStages:
            x = blk(x)

        x = x.view(B, -1, self.dims[-1])
        x = self.norm2(x)                    # input: B L C. return: B L C
        x = self.avgpool(x.transpose(1, 2))  # return: B C 1
        x = torch.flatten(x, 1)
        # x = self.head(x)
        return x

def LNG_T(num_classes=100, config=[1, 1, 2, 1], dim=96, **kwargs):
    return LNGTransformer(in_chans=3, dims=[dim, dim*2, dim*4, dim*8], patch_size=4, window_size=[7, 4], stages=config, num_heads=[3, 6, 12, 24], num_classes=num_classes)

def LNG_S(num_classes=100, config=[1, 1, 2, 1], dim=128, **kwargs):
    return LNGTransformer(in_chans=3, dims=[dim, dim*2, dim*4, dim*8], patch_size=4, window_size=[7, 7], stages=config, num_heads=[4, 8, 16, 32], num_classes=num_classes)

def LNG_B(num_classes=100, config=[1, 1, 6, 1], dim=128, **kwargs):
    return LNGTransformer(in_chans=3, dims=[dim, dim*2, dim*4, dim*8], patch_size=4, window_size=[7, 7], stages=config, num_heads=[4, 8, 16, 32], num_classes=num_classes)

if __name__ == '__main__':
    test_model = LNG_T()
    # print(test_model)
    n_parameters = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
    print(n_parameters)

    # print(test_model)
    dummy_input = torch.rand(1, 3, 224, 128)
    start_time = time.time()
    for i in range(1):
        output = test_model(dummy_input)
    end_time = time.time()
    flops, params = profile(test_model, inputs=(dummy_input,))
    print(end_time-start_time)
    flops, params = clever_format([flops, params], '%.3f')
    print(params)
    print(flops)
