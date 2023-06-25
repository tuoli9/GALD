from functools import partial

import torch
import torch.nn as nn
from timm.models.helpers import *
from timm.models.layers import to_2tuple, trunc_normal_, Mlp
from timm.models.pit import (ConvEmbedding, ConvHeadPooling, SequentialTuple,
                             checkpoint_filter_fn, default_cfgs)
from timm.models.registry import register_model

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, r, attn_guide = None):
        B, N, C = x.shape     #8,197,784  
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)   # 8,12,197,784/12

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)     # bs,4,962,962
        
        if attn_guide != None:
            tradeoff = r  # 0.5最优
            attn = tradeoff * attn + (1-tradeoff) * attn_guide[-1].detach()

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)   
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, r, attn_guide = None):
        temp, attn = self.attn(self.norm1(x), r, attn_guide)
        x = x + self.drop_path(temp)   # bs,197,768
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


def _create_pit(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        PoolingVisionTransformer, variant, pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)
    return model

class Transformer(nn.Module):
    def __init__(
            self, base_dim, depth, heads, mlp_ratio, pool=None, drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None, layer_id = 0):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        embed_dim = base_dim * heads

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_prob[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            for i in range(depth)])

        self.pool = pool
        self.id = layer_id

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], r, attn_guide = None):
        x, cls_tokens = x   # 第一次 x --> bs,256,31,31   bs,512,16,16  bs,1024,8,8
        B, C, H, W = x.shape
        token_length = cls_tokens.shape[1]

        x = x.flatten(2).transpose(1, 2)  
        x = torch.cat((cls_tokens, x), dim=1)   # bs,962,256

        # x = self.blocks(x)
        attns = []
        layer_wise_tokens = []
        for idx, blk in enumerate(self.blocks):
            if attn_guide != None:
                x, attn = blk(x, r, attn_guide[self.id])          # attentions bs,4,962,962  bs,8,257,257  bs,16,65,65
            else:
                x, attn = blk(x,r)
            attns.append(attn)
            layer_wise_tokens.append(x)              # bs,962,256    bs,257,512    bs,65,1024
 
        cls_tokens = x[:, :token_length]   # bs,1,256     bs,1,512       bs,1,1024  
        x = x[:, token_length:]            # bs,961,256   bs,256,512     bs,64,1024
        x = x.transpose(1, 2).reshape(B, C, H, W)   # bs,256,31,31

        if self.pool is not None:
            x, cls_tokens = self.pool(x, cls_tokens)  # bs,512,16,16  bs,1,512    bs,1024,8,8  bs,1,1024  bs,1024,8,8  bs,1,1024
        return (x, cls_tokens), layer_wise_tokens, attns

class PoolingVisionTransformer(nn.Module):
    """ Pooling-based Vision Transformer

    A PyTorch implement of 'Rethinking Spatial Dimensions of Vision Transformers'
        - https://arxiv.org/abs/2103.16302
    """
    def __init__(self, img_size, patch_size, stride, base_dims, depth, heads,
                 mlp_ratio, num_classes=1000, in_chans=3, distilled=False,
                 attn_drop_rate=.0, drop_rate=.0, drop_path_rate=.0):
        super(PoolingVisionTransformer, self).__init__()

        padding = 0
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        height = math.floor((img_size[0] + 2 * padding - patch_size[0]) / stride + 1)
        width = math.floor((img_size[1] + 2 * padding - patch_size[1]) / stride + 1)

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes
        self.num_tokens = 2 if distilled else 1
        self.depth = depth

        self.patch_size = patch_size
        self.pos_embed = nn.Parameter(torch.randn(1, base_dims[0] * heads[0], height, width))
        self.patch_embed = ConvEmbedding(in_chans, base_dims[0] * heads[0], patch_size, stride, padding)

        self.cls_token = nn.Parameter(torch.randn(1, self.num_tokens, base_dims[0] * heads[0]))
        self.pos_drop = nn.Dropout(p=drop_rate)

        transformers = []
        # stochastic depth decay rule
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depth)).split(depth)]
        for stage in range(len(depth)):
            pool = None
            if stage < len(heads) - 1:
                pool = ConvHeadPooling(
                    base_dims[stage] * heads[stage], base_dims[stage + 1] * heads[stage + 1], stride=2)
            transformers += [Transformer(
                base_dims[stage], depth[stage], heads[stage], mlp_ratio, pool=pool,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_prob=dpr[stage], layer_id = stage)
            ]
        self.transformers = SequentialTuple(*transformers)
        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.num_features = self.embed_dim = base_dims[-1] * heads[-1]

        # Classifier head
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        if self.head_dist is not None:
            return self.head, self.head_dist
        else:
            return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.head_dist is not None:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, r = 1.0, attn_guide = None):
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        
        layer_wise_tokens = []
        attns = []
        for layer, transformer in enumerate(self.transformers):
            (x, cls_tokens), layer_wise_token, attn = transformer((x, cls_tokens), r, attn_guide)
            attns.append(attn)
            layer_wise_tokens.append(layer_wise_token)
        # x, cls_tokens = self.transformers((x, cls_tokens))
        cls_tokens = self.norm(cls_tokens)
        if self.head_dist is not None:
            return cls_tokens[:, 0], cls_tokens[:, 1], attns
        else:
            return cls_tokens[:, 0], attns, [self.norm(x[:,:1])[:,0] for x in layer_wise_tokens[-1]]

    def forward(self, x, r = 1.0, attn_guide = None):
        x, attns, layer_wise = self.forward_features(x, r, attn_guide)
        
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            # for i in range(len(tocken_list)):
            #     if i < len(tocken_list) - self.depth[-1]:
            #         layer_wise_tokens.append(tocken_list[i])
            #     else:
            #         layer_wise_tokens.append(self.head(tocken_list[i]))
            return self.head(x), attns, [self.head(y) for y in layer_wise]
  
# @register_model
def pit_b_224(pretrained, **kwargs):
    model_kwargs = dict(
        patch_size=14,
        stride=7,
        base_dims=[64, 64, 64],
        depth=[3, 6, 4],
        heads=[4, 8, 16],
        mlp_ratio=4,
        **kwargs
    )
    return _create_pit('pit_b_224', pretrained, **model_kwargs)


# @register_model
def pit_s_224(pretrained, **kwargs):
    model_kwargs = dict(
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[3, 6, 12],
        mlp_ratio=4,
        **kwargs
    )
    return _create_pit('pit_s_224', pretrained, **model_kwargs)


# @register_model
def pit_xs_224(pretrained, **kwargs):
    model_kwargs = dict(
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs
    )
    return _create_pit('pit_xs_224', pretrained, **model_kwargs)


# @register_model
def pit_ti_224(pretrained, **kwargs):
    model_kwargs = dict(
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs
    )
    return _create_pit('pit_ti_224', pretrained, **model_kwargs)

    
