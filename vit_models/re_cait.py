import torch
import torch.nn as nn
from functools import partial

from timm.models.cait import default_cfgs, checkpoint_filter_fn
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, Mlp, PatchEmbed
from timm.models.helpers import *


def _create_cait(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        Cait, variant, pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)
    return model

class ClassAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA 
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, global_attn = None):
        B, N, C = x.shape          # 8,197,384
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # batchsize, heads, 1, 48
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)   # batchsize, heads, 197, 48

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)   # batchsize, heads, 197, 48

        attn = (q @ k.transpose(-2, -1))   # batchsize, heads, 1,197
        attn = attn.softmax(dim=-1)
        
        if global_attn != None:
            tradeoff = 0.9
            attn = tradeoff * attn + (1-tradeoff) * global_attn
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)   # batchsize, heads,1,48 --> batchsize, 1, 384
        x_cls = self.proj(x_cls)   # 8,1,384
        x_cls = self.proj_drop(x_cls)

        return x_cls, attn


class LayerScaleBlockClassAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_block=ClassAttn,
            mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)  # bs,1,384
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x, x_cls, global_attn = None):        # x_cls-->8,1,384
        u = torch.cat((x_cls, x), dim=1)
        x, attn = self.attn(self.norm1(u), global_attn)
        x_cls = x_cls + self.drop_path(self.gamma_1 * x)
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        return x_cls, attn


class TalkingHeadAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads    # 768 // 8 = 96

        self.scale = head_dim ** -0.5   # 1/根号96

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, global_attn = None):
        B, N, C = x.shape      # 4,196,384
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]    # 4,8,196,48

        attn = (q @ k.transpose(-2, -1))       # 4,8,196,196

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)    

        attn = attn.softmax(dim=-1)       # 4,8,196,196
        if global_attn != None:
            tradeoff = 1.
            attn = tradeoff * attn + (1-tradeoff) * global_attn
        
        soft_attn = attn
        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)   # 4,8,196,48 reshape --> 4,196,384
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, soft_attn

class LayerScaleBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add layerScale
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_block=TalkingHeadAttn,
            mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x, global_attn = None):
        temp, attn = self.attn(self.norm1(x), global_attn)
        x = x + self.drop_path(self.gamma_1 * temp)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x, attn

class Cait(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to adapt to our cait models
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
            num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            global_pool=None,
            block_layers=LayerScaleBlock,
            block_layers_token=LayerScaleBlockClassAttn,
            patch_layer=PatchEmbed,
            act_layer=nn.GELU,
            attn_block=TalkingHeadAttn,
            mlp_block=Mlp,
            init_scale=1e-4,
            attn_block_token_only=ClassAttn,
            mlp_block_token_only=Mlp,
            depth_token_only=2,
            mlp_ratio_clstk=4.0
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = patch_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, attn_block=attn_block, mlp_block=mlp_block, init_values=init_scale)
            for i in range(depth)])

        self.blocks_token_only = nn.ModuleList([
            block_layers_token(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_clstk, qkv_bias=qkv_bias,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer,
                act_layer=act_layer, attn_block=attn_block_token_only,
                mlp_block=mlp_block_token_only, init_values=init_scale)
            for i in range(depth_token_only)])

        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, global_attn = None):
        B = x.shape[0]
        x = self.patch_embed(x)                          # 8,196,384

        cls_tokens = self.cls_token.expand(B, -1, -1)    # 8,1,384

        x = x + self.pos_embed
        x = self.pos_drop(x)

        patch_attns = []
        cls_attns = []
        for i, blk in enumerate(self.blocks):
            if global_attn != None:
                x, patch_attn = blk(x, global_attn[0][-1].detach())
            else:
                x, patch_attn = blk(x)
            patch_attns.append(patch_attn)

        for i, blk in enumerate(self.blocks_token_only):
            if global_attn != None:
                cls_tokens, cls_attn = blk(x, cls_tokens, global_attn[1][-1].detach())
            else:
                cls_tokens, cls_attn = blk(x, cls_tokens)
            cls_attns.append(cls_attn)

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.norm(x)
        return x[:, 0], [patch_attns, cls_attns]

    def forward(self, x, global_attn = None):
        x, attns = self.forward_features(x, global_attn)
        x = self.head(x)
        return x, attns


# class MyCait(Cait):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
    
#     def forward_features(self, x):
#         B = x.shape[0]
#         x = self.patch_embed(x)                          # 8,196,384

#         cls_tokens = self.cls_token.expand(B, -1, -1)    # 8,1,384

#         x = x + self.pos_embed
#         x = self.pos_drop(x)

#         layer_wise_tokens = []
#         for i, blk in enumerate(self.blocks):
#             x,_ = blk(x)
#             # layer_wise_tokens.append(x)

#         for i, blk in enumerate(self.blocks_token_only):
#             cls_tokens,_ = blk(x, cls_tokens)
#             # layer_wise_tokens.append(cls_tokens)
#         # layer_wise_tokens = [self.norm(x) for x in layer_wise_tokens]   # 8, 196, 384
        
#         x = torch.cat((cls_tokens, x), dim=1)

#         x = self.norm(x)
#         return x[:, 0], layer_wise_tokens  # layer_wise_tokens[-1].squeeze(dim = 1) == x[:, 0]  最后两层的cls_tocken都有判别性

#     def forward(self, x):
#         x, layer_wise_tokens = self.forward_features(x)
#         x = self.head(x)
#         # depth_token_only = len(self.blocks_token_only)
#         # for i in range(len(layer_wise_tokens)-depth_token_only,len(layer_wise_tokens)):
#         #     layer_wise_tokens[i] = self.head(layer_wise_tokens[i]).squeeze(1)
#         return x, layer_wise_tokens

# @register_model
def cait_xxs24_224(pretrained=False, **kwargs):
    model_args = dict(patch_size=16, embed_dim=192, depth=24, num_heads=4, init_scale=1e-5, **kwargs)
    model = _create_cait('cait_xxs24_224', pretrained=pretrained, **model_args)
    return model


# @register_model
def cait_xxs24_384(pretrained=False, **kwargs):
    model_args = dict(patch_size=16, embed_dim=192, depth=24, num_heads=4, init_scale=1e-5, **kwargs)
    model = _create_cait('cait_xxs24_384', pretrained=pretrained, **model_args)
    return model


# @register_model
def cait_xxs36_224(pretrained=False, **kwargs):
    model_args = dict(patch_size=16, embed_dim=192, depth=36, num_heads=4, init_scale=1e-5, **kwargs)
    model = _create_cait('cait_xxs36_224', pretrained=pretrained, **model_args)
    return model


# @register_model
def cait_xxs36_384(pretrained=False, **kwargs):
    model_args = dict(patch_size=16, embed_dim=192, depth=36, num_heads=4, init_scale=1e-5, **kwargs)
    model = _create_cait('cait_xxs36_384', pretrained=pretrained, **model_args)
    return model


# @register_model
def cait_xs24_384(pretrained=False, **kwargs):
    model_args = dict(patch_size=16, embed_dim=288, depth=24, num_heads=6, init_scale=1e-5, **kwargs)
    model = _create_cait('cait_xs24_384', pretrained=pretrained, **model_args)
    return model


# @register_model
def cait_s24_224(pretrained=False, **kwargs):
    model_args = dict(patch_size=16, embed_dim=384, depth=24, num_heads=8, init_scale=1e-5, **kwargs)
    model = _create_cait('cait_s24_224', pretrained=pretrained, **model_args)
    return model


# @register_model
def cait_s24_384(pretrained=False, **kwargs):
    model_args = dict(patch_size=16, embed_dim=384, depth=24, num_heads=8, init_scale=1e-5, **kwargs)
    model = _create_cait('cait_s24_384', pretrained=pretrained, **model_args)
    return model


# @register_model
def cait_s36_384(pretrained=False, **kwargs):
    model_args = dict(patch_size=16, embed_dim=384, depth=36, num_heads=8, init_scale=1e-6, **kwargs)
    model = _create_cait('cait_s36_384', pretrained=pretrained, **model_args)
    return model


# @register_model
def cait_m36_384(pretrained=False, **kwargs):
    model_args = dict(patch_size=16, embed_dim=768, depth=36, num_heads=16, init_scale=1e-6, **kwargs)
    model = _create_cait('cait_m36_384', pretrained=pretrained, **model_args)
    return model


# @register_model
def cait_m48_448(pretrained=False, **kwargs):
    model_args = dict(patch_size=16, embed_dim=768, depth=48, num_heads=16, init_scale=1e-6, **kwargs)
    model = _create_cait('cait_m48_448', pretrained=pretrained, **model_args)
    return model