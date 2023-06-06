import gc, math

from inspect import isfunction

import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import checkpoint

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

        self.gamma_ff = nn.Parameter(torch.ones(dim_out))

    def forward(self, x):
        return self.gamma_ff * self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64,
                 dropout=0.0, use_dropout2d=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout2d(dropout) if use_dropout2d else nn.Dropout(dropout)
        )

        self.gamma_q = nn.Parameter(torch.ones(inner_dim))
        self.gamma_k = nn.Parameter(torch.ones(inner_dim))
        self.gamma_v = nn.Parameter(torch.ones(inner_dim))
        self.gamma_out = nn.Parameter(torch.ones(inner_dim))

        # Valid values for steps = (2, 4, 8, 16, 32, 64).
        # Higher steps is slower but less memory usage.
        # At 16 we can run 1920x1536 on a 3090, at 64 we
        # can run over 1920x1920. Speed seems to be
        # impacted more on 30x series cards.
        self.attn_steps = 4

    def _sliced_attn(self, q, k, v):
        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device)
        #r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device='cpu')
        slice_size = q.shape[1] // self.attn_steps if q.shape[1] % self.attn_steps == 0 else q.shape[1]
        for i in range(0, q.shape[1], slice_size):
            end = i + slice_size
            #s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k)
            #s1 *= self.scale

            s1 = torch.baddbmm(
                torch.empty(q.shape[0], slice_size, k.shape[1], dtype=q.dtype, device=q.device),
                q[:, i:end],
                k.transpose(-1, -2),
                beta=0,
                alpha=self.scale,
            )

            s2 = s1.softmax(dim=-1)
            del s1

            #r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)

            r1[:, i:end] = torch.bmm(s2, v)
            #r1[:, i:end] = torch.bmm(s2, v).cpu()
            del s2

        return r1

    def forward(self, x, context=None, mask=None):
        h = self.heads
        #b, n, hd = x.shape

        q_in = self.gamma_q * self.to_q(x)
        context = default(context, x)
        k_in = self.gamma_k * self.to_k(context)
        v_in = self.gamma_v * self.to_v(context)
        del context, x

        #q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        rearrange_qkv = lambda y: rearrange(y, 'b n (h d) -> (b h) n d', h=self.heads)
        #rearrange_qkv = lambda y: y.view(b * h, -1, hd // h)
        q = rearrange_qkv(q_in)
        del q_in
        k = rearrange_qkv(k_in)
        del k_in
        v = rearrange_qkv(v_in)
        del v_in

        r2 = None
        while r2 is None:
            try:
                r1 = self._sliced_attn(q, k, v)
                r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
                del r1
            except RuntimeError as e:
                if self.attn_steps < 64:
                    self.attn_steps *= 2

                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    print(f'Increased attn_steps to {self.attn_steps} without success - reraising OutOfMemoryException')
                    raise e

        #r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device)

        #steps = 16
        #slice_size = q.shape[1] // steps if q.shape[1] % steps == 0 else q.shape[1]
        #for i in range(0, q.shape[1], slice_size):
        #    end = i + slice_size
        #    #s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k)
        #    #s1 *= self.scale

        #    s1 = torch.baddbmm(
        #        torch.empty(q.shape[0], slice_size, k.shape[1], dtype=q.dtype, device=q.device),
        #        q[:, i:end],
        #        k.transpose(-1, -2),
        #        beta=0,
        #        alpha=self.scale,
        #    )

        #    s2 = s1.softmax(dim=-1)
        #    del s1

        #    #r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
        #    r1[:, i:end] = torch.bmm(s2, v)
        #    del s2

        to_out, out_dropout = self.to_out[0], self.to_out[1]
        out = self.gamma_out * to_out(r2)
        return out_dropout(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64,
                 dropout=0.0, use_dropout2d=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout2d(dropout) if use_dropout2d else nn.Dropout(dropout)
        )
        self.attention_op = None

        self.gamma_q = nn.Parameter(torch.ones(inner_dim))
        self.gamma_k = nn.Parameter(torch.ones(inner_dim))
        self.gamma_v = nn.Parameter(torch.ones(inner_dim))
        self.gamma_out = nn.Parameter(torch.ones(inner_dim))

    def forward(self, x, context=None, mask=None, upcast_attn=True, **kwargs):
        # https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/8367/files#diff-477a645246ea31dd6f7fc79f64aef19e8dce7772116d0885cdb8d0c438a1bedf

        h = self.heads
        context = default(context, x)
        batch_size, sequence_length, inner_dim = x.shape

        if mask is not None:
            raise Exception('Attention masking not implemented')

        q_in = self.gamma_q * self.to_q(x)
        k_in = self.gamma_k * self.to_k(context)
        v_in = self.gamma_v * self.to_v(context)

        head_dim = inner_dim // h
        q = q_in.view(batch_size, -1, h, head_dim).transpose(1, 2)
        k = k_in.view(batch_size, -1, h, head_dim).transpose(1, 2)
        v = v_in.view(batch_size, -1, h, head_dim).transpose(1, 2)

        del q_in, k_in, v_in

        dtype = q.dtype
        if upcast_attn:
            q, k = q.float(), k.float()

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):
            hidden_states = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, h * head_dim)
        hidden_states = hidden_states.to(dtype)

        # linear proj
        hidden_states = self.gamma_out * self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states

    def xformers_forward(self, x, context=None):
        q = self.gamma_q * self.to_q(x)
        context = default(context, x)
        k = self.gamma_k * self.to_k(context)
        v = self.gamma_v * self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .half()
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v,
                                                      op=self.attention_op)

        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
            #.float()
        )

        to_out, out_dropout = self.to_out[0], self.to_out[1]
        out = self.gamma_out * to_out.half()(out)
        return out_dropout(out)
        #return self.to_out.half()(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,
        "softmax-xformers": MemoryEfficientCrossAttention
    }

    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None,
                 gated_ff=True, use_checkpoint=True, disable_flash_attn=False,
                 use_dropout2d=False, dropout_only_xattn=False):
        super().__init__()

        if disable_flash_attn:
            attn_mode = 'softmax'
        else:
            attn_mode = 'softmax-xformers'

        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.attn1 = attn_cls(
            query_dim=dim, heads=n_heads, dim_head=d_head,
            dropout=0.0 if dropout_only_xattn else dropout, use_dropout2d=use_dropout2d
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=0.0 if dropout_only_xattn else dropout, glu=gated_ff)
        self.attn2 = attn_cls(
            query_dim=dim, context_dim=context_dim,
            heads=n_heads, dim_head=d_head,
            dropout=dropout, use_dropout2d=use_dropout2d
        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.use_checkpoint = use_checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.use_checkpoint)

    def _forward(self, x_0, context=None):
        x_1 = x_0 + self.attn1(self.norm1(x_0))
        del x_0
        if context is None:
            x_2 = x_1 + self.norm2(x_1)
        else:
            x_2 = x_1 + self.attn2(self.norm2(x_1), context=context)
        del x_1
        x_out = x_2 + self.ff(self.norm3(x_2))
        del x_2
        return x_out


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, use_checkpoint=True,
                 disable_flash_attn=False, use_dropout2d=False,
                 dropout_only_xattn=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim, n_heads, d_head,
                dropout=dropout, context_dim=context_dim,
                use_checkpoint=use_checkpoint,
                disable_flash_attn=disable_flash_attn,
                use_dropout2d=use_dropout2d,
                dropout_only_xattn=dropout_only_xattn)
            for d in range(depth)
        ])

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

        #self.norm_out = torch.nn.InstanceNorm2d(in_channels, affine=True)
        #self.norm_out = Normalize(in_channels)

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        #x = self.norm_out(x)
        return x + x_in
