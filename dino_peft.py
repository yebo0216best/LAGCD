import math
from operator import mul
from functools import reduce
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F

import copy

class VPT(nn.Module):
    def __init__(self, vpt_len, seq_len, patch_size, emb_dim, dtype=None):
        super().__init__()
        self.seq_len = seq_len
        self.prompt = nn.Parameter(torch.empty(vpt_len, emb_dim, dtype=dtype))
        init_val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + emb_dim))
        nn.init.uniform_(self.prompt, -init_val, init_val)
    def forward(self, x):
        x = x[:, :self.seq_len, :]
        prompt = self.prompt.expand(x.shape[0], -1, -1)
        x = torch.cat([x, prompt], dim=1)
        return x

class ReluAdapter(nn.Module):
    def __init__(self, feat_dim, mid_dim, scale, dtype=None, dropout_prob=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(feat_dim, dtype=dtype)
        self.down_proj = nn.Linear(feat_dim, mid_dim, dtype=dtype)
        self.up_proj = nn.Linear(mid_dim, feat_dim, dtype=dtype)
        self.scale = scale
        self.dropout_prob = dropout_prob
        self.activation = nn.ReLU(inplace=True)
        # self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # self.activation = nn.Threshold(threshold=1.0, value=0.0)
        # self.activation = nn.ELU(alpha=1.0, inplace=True)
        # self.activation = nn.SiLU(inplace=True)
        # self.activation = nn.GELU(approximate='none')

        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    @property
    def dtype(self):
        return self.down_proj.weight.dtype

    def forward(self, x):
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.activation(x)
        x = nn.functional.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.up_proj(x)
        x = x * self.scale
        return x

class LinearAdapter(nn.Module):
    def __init__(self, feat_dim, mid_dim, scale, dtype=None, dropout_prob=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(feat_dim, dtype=dtype)
        self.down_proj = nn.Linear(feat_dim, mid_dim, dtype=dtype)
        self.up_proj = nn.Linear(mid_dim, feat_dim, dtype=dtype)
        self.scale = scale
        self.dropout_prob = dropout_prob

        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    @property
    def dtype(self):
        return self.down_proj.weight.dtype

    def forward(self, x):
        x = self.ln(x)
        x = self.down_proj(x)
        x = nn.functional.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.up_proj(x)
        x = x * self.scale
        return x

class ViT_Tuner(nn.Module):
    def __init__(self, dino_model, args):
        super().__init__()
        n_layers = dino_model.depth
        patch_size = dino_model.patch_embed.proj.kernel_size
        dtype = dino_model.patch_embed.proj.weight.dtype
        feat_dim = 768
        seq_len = 197
        partial = args.partial
        mid_dim = args.mid_dim
        scale = args.adapter_scale
        dropout_prob = args.dropout_prob

        # Selecting the fine-tuning method
        if args.use_gcdtune:
            gcdtune_list = nn.ParameterList([
                param for name, param in dino_model.named_parameters()
                if "block" in name and int(name.split(".")[1]) == 11])
        else:
            gcdtune_list = None

        vpt_len = args.prompt_len
        if args.use_vpt:
            vpt_list = nn.ModuleList([
                *[None] * (n_layers - partial),
                *[VPT(vpt_len=vpt_len, seq_len=seq_len, patch_size=patch_size, emb_dim=feat_dim, dtype=dtype) for _ in
                  range(partial)]
            ])
        else:
            vpt_list = nn.ModuleList([None] * n_layers)

        if args.use_relu:
            relu_list = nn.ModuleList([
                *[None] * (n_layers - partial),
                *[ReluAdapter(feat_dim=feat_dim, mid_dim=mid_dim, scale=scale, dtype=dtype, dropout_prob=dropout_prob) for _ in range(partial)]
            ])
        else:
            relu_list = nn.ModuleList([None] * n_layers)

        if args.use_linear:
            linear_list = nn.ModuleList([
                *[None] * (n_layers - partial),
                *[LinearAdapter(feat_dim=feat_dim, mid_dim=mid_dim, scale=scale, dtype=dtype, dropout_prob=dropout_prob) for _ in range(partial)]
            ])
        else:
            linear_list = nn.ModuleList([None] * n_layers)

        # To be optimized
        self.gcdtune_list = gcdtune_list
        self.vpt_list = vpt_list
        self.relu_list = relu_list
        self.linear_list = linear_list


class DINO_ViT(nn.Module):
    def __init__(self, dino_model):
        super().__init__()
        self.interpolate_pos_encoding = dino_model.interpolate_pos_encoding
        self.patch_embed = dino_model.patch_embed
        self.cls_token = dino_model.cls_token
        self.pos_embed = dino_model.pos_embed
        self.pos_drop = dino_model.pos_drop
        self.blocks = dino_model.blocks
        self.norm = dino_model.norm
        self.depth = dino_model.depth

    def forward(self, x, tuner=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)
        _bsz = x.shape[0]
        _seq_len = x.shape[1]
        _emb_dim = x.shape[2]
        n_layers = self.depth

        for i in range(n_layers):
            block = self.blocks[i]
            if tuner is not None:
                vpt = tuner.vpt_list[i]
                relu_adapter = tuner.relu_list[i]
                linear_adapter = tuner.linear_list[i]
            else:
                vpt = None
                relu_adapter = None
                linear_adapter = None

            if vpt is not None:
                x = vpt(x)
            # NLD -> LND
            _seq_len_after_vpt = x.shape[1]
            x = x.permute(1, 0, 2)
            # Decomposed Block Weight
            _attn = block.attn
            _ln_1 = block.norm1
            _mlp = block.mlp
            _ln_2 = block.norm2
            _attn_in_proj_weight = _attn.qkv.weight
            _attn_in_proj_bias = _attn.qkv.bias
            _attn_out_proj_weight = _attn.proj.weight
            _attn_out_proj_bias = _attn.proj.bias
            _mlp_in_proj_weight = _mlp.fc1.weight
            _mlp_in_proj_bias = _mlp.fc1.bias
            _mlp_act = _mlp.act
            _mlp_out_proj_weight = _mlp.fc2.weight
            _mlp_out_proj_bias = _mlp.fc2.bias
            _num_heads = _attn.num_heads
            _head_dim = _emb_dim // _num_heads

            ###############################
            ## Multi-Head Self-Attention ##
            ###############################
            identity_attn = x  # deep copy
            x = _ln_1(x)
            qkv = F.linear(x, _attn_in_proj_weight, _attn_in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.contiguous().view(q.shape[0], q.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            k = k.contiguous().view(k.shape[0], k.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            v = v.contiguous().view(v.shape[0], v.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            # scaled_dot_product_attention:
            q = q / math.sqrt(_head_dim)
            attn = torch.bmm(q, k.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1)
            x = torch.bmm(attn, v)
            x = x.transpose(0, 1).contiguous().view(-1, _emb_dim)
            x = F.linear(x, _attn_out_proj_weight, _attn_out_proj_bias)
            x = x.view(_seq_len_after_vpt, _bsz, _emb_dim)
            x = x + identity_attn

            ##########################
            ## Feed-Forward Network ##
            ##########################
            identity_ffn = x  # deep copy
            x = _ln_2(x)
            x_out = F.linear(x, _mlp_in_proj_weight, _mlp_in_proj_bias)
            x = x_out
            x = _mlp_act(x)
            x_out = F.linear(x, _mlp_out_proj_weight, _mlp_out_proj_bias)
            x = x_out

            # Applying adapter to FFN
            if linear_adapter is not None:
                x = x + linear_adapter(identity_ffn)
            if relu_adapter is not None:
                x = x + relu_adapter(identity_ffn)

            x = x + identity_ffn
            x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.norm(x)

        return x[:, 0]

class Dino_Model(nn.Module):
    def __init__(self, dino_model, args):
        super().__init__()
        self.image_encoder = DINO_ViT(dino_model)
        self.tuner = ViT_Tuner(dino_model, args)

    def forward(self, image):
        tuner = self.tuner
        feat = self.image_encoder(image, tuner)
        return feat



