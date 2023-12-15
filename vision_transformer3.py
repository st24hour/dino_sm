# ------------------------------------------------------------------------
# CL3IP re-implementation code of ViT : model/vision_transformer.py
# Copyright (c) LG AI Research, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch
import torch.nn as nn

# from timm.models.vision_transformer import PatchEmbed, Block
from vision_misc_ import PatchEmbed, Block
from timm.models.layers import trunc_normal_, lecun_normal_

# js
import math

class VisionTransformer3(nn.Module):
    """ Random Masked Vision Transformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0,
                 embed_dim=768, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 drop_rate=0., drop_path_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        act_layer = nn.GELU
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # --------------------------------------------------------------------------
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) # original
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.depth = depth

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        
        self.norm_pre = norm_layer(embed_dim)
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # Classifier head(s)
        self.pre_logits = nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # self.head = nn.Identity()

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        trunc_normal_(self.pos_embed, std=.02)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio, return_ids, fixed_ids_keep):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        if fixed_ids_keep:
            ids_keep = fixed_ids_keep
        else:
            N, L, D = x.shape  # batch, length, dim
            len_keep = int(L * (1 - mask_ratio))
            
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
            
            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

            # keep the first subset
            ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        if return_ids:
            return x_masked, ids_keep

        return x_masked

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        # npatch = x.shape[1]
        # N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
    def forward_features(self, x, mask_ratio, return_ids, ids_keep):
        B, nc, w, h = x.shape
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        # x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if mask_ratio > 0:
            x = self.random_masking(x, mask_ratio, return_ids, ids_keep)
            if return_ids: x, ids_keep = x

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)
        x = self.norm_pre(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if return_ids: self.pre_logits(x[:,0]), ids_keep
        return self.pre_logits(x[:, 0])

    def forward(self, imgs, mask_ratio=0., return_ids=False, ids_keep=None):
        x = self.forward_features(imgs, mask_ratio, return_ids, ids_keep)
        if return_ids: x, ids_keep = x
        x = self.head(x) # runs here

        if return_ids: return x, ids_keep
        return x


##########################################################################################
# model call
# ViT-B/32, B/16, L/14
def vision_transformer3(**kwargs):
    assert kwargs['mode'] in ['base', 'large'], "Only ViT-B, ViT-L is supported."
    vision_kwargs = {
        'img_size': 224,
        'patch_size': kwargs['patch_size'],
        'num_classes': kwargs['num_classes'] if 'num_classes' in kwargs else 0,
        'embed_dim': 1024 if kwargs['mode'] == 'large' else 768,
        'num_heads': 16 if kwargs['mode'] == 'large' else 12,
        'depth': 24 if kwargs['mode'] == 'large' else 12,
        'mlp_ratio': 4,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
    }
    model = VisionTransformer3(**vision_kwargs)

    return model