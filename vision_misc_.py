from torch import _assert
from timm.layers.helpers import to_2tuple
from timm.models.layers import trunc_normal_, lecun_normal_, Mlp, DropPath
from timm.models.vision_transformer import default_cfgs
import torch.nn as nn
import torch

########################################################################################################
# Transformer functions
########################################################################################################
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=False,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        elif 'pre_logits' in k:
            # NOTE representation layer removed as not used in latest 21k/1k pretrained weights
            continue
        out_dict[k] = v
    return out_dict


default_cfgs.update({
    'vit_large_patch14_224': {
        'url': '',
        'num_classes': 0,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': 1.0,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        'first_conv': 'patch_embed.proj',
        'classifier': 'head'
    }
})


################################################################################################
# Attention
################################################################################################
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., keep_rate=1., use_fuse=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_fuse = use_fuse
        self.keep_rate = keep_rate
        assert 0 < keep_rate <= 1, "keep_rate must be > 0 and <= 1, got {0}".format(keep_rate)

    def forward(self, x, keep_rate=None, distilled=False):
        if keep_rate is None: keep_rate = self.keep_rate
        B, N, C = x.shape # ViT-B/16 : ([B, 1+16*16, 768])
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        # H = 12, transformer_dim=64, q,k,v = ([batch, depth, 1+16*16, 64])

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # attn = ([B, H, 197, 197])

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # attn @ v = ([B, H, 197, 197]) @ ([B, H, 197, 64]) = ([B, H, 197, 64])
        x = self.proj(x)
        x = self.proj_drop(x)
        # x = [B, 197, 768]

        # borrowed expedition from EViT (https://github.com/youweiliang/evit)
        num_tokens_left = (N-2) if distilled else (N-1)
        if self.keep_rate < 1 and keep_rate < 1:
            num_tokens_left = math.ceil(keep_rate * (N-1))
            if num_tokens_left == N-1:
                return x, None, None, None
            assert num_tokens_left >= 1

            if distilled:
                cls_attn = attn[:, :, 0, 2:] # attn between [CLS] token and the rest, [B, H, N-1]
            else:
                cls_attn = attn[:, :, 0, 1:] # attn between [CLS] token and the rest, [B, H, N-1]
            cls_attn = cls_attn.mean(dim=1) # [B, N-1], average attn throughout the heads
            
            _, idx = torch.topk(cls_attn, num_tokens_left, dim=1, largest=True, sorted=True)
            index = idx[:, :num_tokens_left].unsqueeze(-1).expand(-1, -1, C) # [B, num_tokens_left, 768]
            non_topk_index = idx[:, num_tokens_left:].unsqueeze(-1).expand(-1, -1, C) # [B, N-1-num_tokens_left, 768] exclude fuse token when fusing
            non_topk_attn = torch.gather(cls_attn, dim=1, index=idx[:, num_tokens_left:]).softmax(dim=1)

            return x, index, non_topk_index, non_topk_attn

        return x, None, None, None


################################################################################################
# Block
################################################################################################
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_rate=1., use_fuse=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, keep_rate=keep_rate, use_fuse=use_fuse)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.keep_rate = keep_rate
        self.use_fuse = use_fuse

    def forward(self, x, keep_rate=None, distilled=False):
        if keep_rate is None: keep_rate = self.keep_rate
        # tmp, index, non_topk_index, non_topk_attn, ids_restore = self.attn(self.norm1(x), keep_rate)
        tmp, index, non_topk_index, non_topk_attn = self.attn(self.norm1(x), keep_rate, distilled)
        x = x + self.drop_path(tmp)

        # borrowed expedition from EViT (https://github.com/youweiliang/evit)
        if index is not None:
            if distilled:
                non_cls_tokens = x[:, 2:] # exclude [CLS], [dist] token
            else:
                non_cls_tokens = x[:, 1:] # exclude [CLS] token
            x_others = torch.gather(non_cls_tokens, dim=1, index=index)

            if self.use_fuse:
                non_topk = torch.gather(non_cls_tokens, dim=1, index=non_topk_index)
                extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)
                if distilled:
                    x = torch.cat([x[:, 0:2], x_others, extra_token], dim=1) # 1 + [0.66 * vis_token] + 1
                else:
                    x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1) # 1 + [0.66 * vis_token] + 1
            else:
                if distilled:
                    x = torch.cat([x[:, 0:2], x_others], dim=1)
                else:
                    x = torch.cat([x[:, 0:1], x_others], dim=1)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # if index is not None: return x, aug1_embed, aug2_embed
        # return x, None, None
        return x