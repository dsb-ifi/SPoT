import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from .utils import (
    gaussian_mixture_sampler_2d, sample_mae_mask, apply_mae_mask,
    _gms_get_grid, _gms_get_gaussian_mask, _gms_sample_mus, 
    _gms_get_gaussian_pos
)


class SelectiveLinearWeightUpdateHook:

    '''A backward hook for a nn.Linear layer to mask input weights from updating.
    '''

    def __init__(self, channels, ksize):
        self.weight_slice = slice(None, channels*ksize**2)

    def __call__(self, weight_grad):  
        # Set gradients for the masked weights to zero        
        if weight_grad is not None:
            weight_grad = weight_grad.clone()
            weight_grad[:, self.weight_slice].zero_()
            return weight_grad
        return weight_grad
    

class RandomGaussianTokenExtractor(nn.Module):

    _valid_samplers = ['uniform', 'gauss', 'sobol', 'grid_uniform', 'grid_center', 'uniform_perturb', 'gauss_perturb', 'edge']
    
    def __init__(
        self, n_features, ksize, n_sigma, 
        sampler='uniform', logprior=None, 
        learnable_n_sigma=False, learnable_logprior=False, **kwargs
    ):
        super().__init__()
        self.n_features = n_features
        self.ksize = ksize
        assert sampler in self._valid_samplers, f'Invalid sampler: {sampler}.'
        self.sampler = sampler
        self._mu = None
        if logprior is not None:
            if learnable_logprior:
                self.logprior = nn.Parameter(torch.tensor(logprior))
            else:
                self.register_buffer('logprior', torch.tensor(logprior), persistent=False)
        else:
            self.register_buffer('logprior', None, persistent=False)
        
        self.register_buffer('grid', _gms_get_grid(ksize), persistent=False)

        if learnable_n_sigma:
            self._n_sigma = nn.Parameter(self._init_n_sigma(n_sigma))
            self.register_buffer('_gaussian', None, persistent=False)
        else:
            self.register_buffer('_n_sigma', self._init_n_sigma(n_sigma))
            self.register_buffer('_gaussian', _gms_get_gaussian_mask(self.grid, n_sigma), persistent=False)
            

    @staticmethod
    def _init_n_sigma(n_sigma):
        t = torch.ones(1)*n_sigma
        return t.exp().sub(1).log() # Softplus inverse
    
    @property
    def sampler(self):
        return self._sampler
    
    @sampler.setter
    def sampler(self, val):
        if val not in self._valid_samplers:
            raise ValueError(f'Invalid sampler: {val}.')
        self._sampler = val
    
    @property
    def n_sigma(self):
        return F.softplus(self._n_sigma)
    
    @property
    def gaussian(self):
        if self._gaussian is None:
            return _gms_get_gaussian_mask(self.grid, self.n_sigma)
        else:
            return self._gaussian

    @property
    def mu(self):
        mu = self._mu
        self._mu = None
        return mu
    
    @mu.setter
    def mu(self, val):
        self._mu = val

    def get_sigma(self, img):
        batch_size, channels, height, width = img.shape
        prec = 1/self.n_sigma
        return self.ksize / (prec*height + prec*width)

    @property
    def patch_info_ratio(self):
        '''Ratio of information per patch w.r.t. masking.
        '''
        return self.gaussian.sum() / self.ksize**2        
        
    def forward(self, img, keep_mu=False):
        batch_size, _, height, width = img.shape
        mu = _gms_sample_mus(
            batch_size, self.n_features, self.sampler, self.grid.device, 
            self.ksize, (height, width)
        )
        if keep_mu:
            self.mu = mu.detach()
        features = gaussian_mixture_sampler_2d(
            img, self.n_features, ksize=self.ksize, n_sigma=self.n_sigma,
            grid=self.grid, mu=mu, logprior=self.logprior
        )
        B, N, C, H, W = features.shape
        return features.view(B, N, -1)
    

class TokenEmbedder(nn.Module):
    
    def __init__(self, ksize, embed_dim, channels=4, pretrain_pos=False, **kwargs):
        super().__init__()
        self.channels = channels
        self.ksize = ksize
        self.embed_dim = embed_dim
        self.input_dim = ksize**2 * channels
        self.emb = nn.Linear(self.input_dim, self.embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 1e-6)
        self._pthook = None
        if pretrain_pos:
            self._pthook = self.emb.weight.register_hook(
                SelectiveLinearWeightUpdateHook(channels-1, ksize)
            )

    def remove_pthook(self):
        if self._pthook is not None:
            self._pthook.remove()
    
    def forward(self, x, *args):
        B, N, E = x.shape
        embedding = self.emb(x)
        return torch.cat([self.cls_token.expand(B, 1, -1), embedding], 1)


class IdentityTokenizer:

    def __call__(self, x, *args, **kwargs):
        return x

    
class MAEDecoderEmbedder(nn.Module):
    
    def __init__(self, ksize, embed_dim, encoder_embed_dim, channels=4):
        super().__init__()
        self.channels = channels
        self.ksize = ksize
        self.embed_dim = embed_dim
        self.emb = nn.Linear(encoder_embed_dim, embed_dim)
        self.pos_emb = nn.Linear(ksize**2, embed_dim, bias=False)
        self.mask_token = nn.Parameter(torch.randn(1, embed_dim) * 1e-6)
        self.cls_pos_emb = nn.Parameter(torch.randn(1, 1, embed_dim) * 1e-6)
    
    def forward(self, xh, x, mask):
        # Get dimensions
        B, N, _ = x.shape
        E = self.embed_dim
        C, H, W = self.channels, self.ksize, self.ksize
        
        # Extract positional embeddings
        pos_embeddings = torch.cat([
            self.cls_pos_emb.expand(B, 1, -1),
            self.pos_emb(x.view(B, N, C, H, W)[:,:,-1].view(B, N, -1))
        ], 1)

        # Construct outputs, accounting for cls_tokens
        out = torch.zeros(B, N+1, E, device=xh.device, dtype=xh.dtype)
        n_dropped = (~mask).sum().item()
        
        # Add projected unmasked tokens to outputs
        out[mask] = self.emb(xh).view(-1, E).to(out.dtype)
        
        # Add learnable mask tokens to outputs
        out[~mask] = self.mask_token.expand(n_dropped, E).to(out.dtype)

        return out + pos_embeddings
    
    
class MSA(nn.Module):
    
    def __init__(
        self, embed_dim:int, heads:int, 
        dop_att:float=0.0, dop_proj:float=0.0, qkv_bias:bool=False,
        lnqk:bool=False, **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.scale = self.head_dim ** -.5
        self.dop_att = dop_att
        self.dop_proj = dop_proj
        self.qkv = nn.Linear(embed_dim, 3*embed_dim, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        if lnqk:
            self.ln_k = nn.LayerNorm(self.head_dim, eps=1e-6)
            self.ln_q = nn.LayerNorm(self.head_dim, eps=1e-6)
        else:
            self.ln_k = nn.Identity()
            self.ln_q = nn.Identity()
            
    def doo(self, x):
        return F.dropout(x, self.dop_proj, training=self.training)

    def doa(self, x):
        return F.dropout(x, self.dop_att, training=self.training)    
        
    def forward(self, features):
        B, N, E = features.shape
        H, D = self.heads, self.head_dim
        q, k, v = (
            self.qkv(features)
                .view(B, N, 3, H, D)
                .permute(2,0,3,1,4)
        )

        feat = F.scaled_dot_product_attention(
            self.ln_q(q), self.ln_k(k), v
        ).transpose(1,2).reshape(B, N, E)

        feat = self.proj(feat)
        
        return self.doo(feat)


class MLP(nn.Module):
    
    def __init__(self, embed_dim, hid_dim, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.act = nn.GELU()
        self.L1 = nn.Linear(embed_dim, hid_dim)
        self.L2 = nn.Linear(hid_dim, embed_dim)
    
    def forward(self, x):
        x = self.act(self.L1(x))
        return self.L2(x)    

    
class LayerScale(nn.Module):

    def __init__(self, embed_dim:int, init_val:float=1e-5):
        super().__init__()
        self.lambda_ = nn.Parameter(torch.full((embed_dim,), init_val))

    def forward(self, x):
        return x * self.lambda_


class DropPath(nn.Module):

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.p = drop_prob
        self.q = 1 - drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.p == 0 or not self.training:
            return x
        
        shape = (x.size(0), *((1,)*(x.ndim-1)))
        drops = x.new_empty(*shape).bernoulli_(self.q)
        
        if self.q > 0. and self.scale_by_keep:
            drops.div_(self.q)

        return x * drops
    
    
class ViTBlock(nn.Module):

    def __init__(
        self, embed_dim, heads, mlp_ratio=4.0, dop_path:float=0.0, **kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ls1 = LayerScale(embed_dim)
        self.ls2 = LayerScale(embed_dim)
        self.dop1 = DropPath(dop_path)
        self.dop2 = DropPath(dop_path)
        hid_dim = int(embed_dim * mlp_ratio)
        self.att = MSA(embed_dim, heads, **kwargs)
        self.mlp = MLP(embed_dim, hid_dim)

    def forward(self, x):
        x = x + self.dop1(self.ls1(self.att(self.norm1(x))))
        x = x + self.dop2(self.ls2(self.mlp(self.norm2(x))))
        return x
    
    

class SPoT(nn.Module):
    
    def __init__(
        self, embed_dim, heads, depth, n_features=256, 
        ksize=24, n_sigma=1.5, in_channels=3, pre_norm=False, mae_drop=0.75,
        decoder=False, encoder_embed_dim=768, **kwargs
    ):
        super().__init__()
        
        if decoder:
            self.tokenizer = IdentityTokenizer()
            self.embedder = MAEDecoderEmbedder(
                ksize, embed_dim, encoder_embed_dim, in_channels+1
            )
        else:
            self.tokenizer = RandomGaussianTokenExtractor(
                n_features, ksize, n_sigma, **kwargs
            )
            self.embedder = TokenEmbedder(ksize, embed_dim, in_channels+1, **kwargs)
            
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, heads, **kwargs)
            for _ in range(depth)
        ])
        self.mae_drop = mae_drop
        self.prenorm = nn.LayerNorm(embed_dim, eps=1e-6) if pre_norm else nn.Identity()
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def mae_encoder_preprocess(self, x):
        if isinstance(self.tokenizer, RandomGaussianTokenExtractor):
            B, K, C = x.shape[0], self.tokenizer.ksize, self.embedder.channels
            mask = sample_mae_mask(x, self.mae_drop)
            z = apply_mae_mask(x, mask, self.mae_drop)
            z.view(B, -1, C, K, K)[:,:,:C-1].mul_(self.tokenizer.gaussian)
            return z, mask
        return None, None
    
    def forward(self, x, *args, mae_encode=False, keep_mu=False):
        x = self.tokenizer(x, keep_mu)
        z = x
        mask = None
        if mae_encode:
            z, mask = self.mae_encoder_preprocess(x)
        
        z = self.prenorm(self.embedder(z, *args))
        
        for block in self.blocks:
            z = block(z)
        
        if mask is not None:
            # Add column to account for cls tokens
            mask = F.pad(mask, (1,0), value=True)
            return self.norm(z), x, mask
        
        return self.norm(z)


class SPoTClassifier(SPoT):

    def __init__(
        self, embed_dim, heads, depth, n_classes, n_features=256,
        ksize=16, n_sigma=1.0, in_channels=3, pre_norm=False, 
        global_pool=False, **kwargs
    ):
        super().__init__(
            embed_dim, heads, depth, n_features=n_features,
            ksize=ksize, n_sigma=n_sigma, in_channels=in_channels, 
            pre_norm=pre_norm, mae_drop=0.0, 
            **kwargs
        )
        self.global_pool = global_pool
        if global_pool:
            self.norm = nn.Identity()
            self.fc_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def custom_mu_tokenize(self, x, mu):
        assert isinstance(self.tokenizer, RandomGaussianTokenExtractor)
        n_features = mu.shape[1]
        B = x.shape[0]
        mu = mu.expand(B, n_features, 1, 1, 2)
        features = gaussian_mixture_sampler_2d(
            x, n_features, ksize=self.tokenizer.ksize, n_sigma=self.tokenizer.n_sigma,
            grid=self.tokenizer.grid, mu=mu, logprior=self.tokenizer.logprior
        )
        B, N, C, H, W = features.shape
        return features.view(B, N, -1)

    def forward(self, x, keep_mu=False, mu=None, cls_token=False, all_tokens=False):
        if mu is None:
            x = super().forward(x, keep_mu=keep_mu)
        else:
            # Branch for conditioning on custom mus
            x = self.custom_mu_tokenize(x, mu)
            x = self.prenorm(self.embedder(x))
            
            for block in self.blocks:
                x = block(x) 
            
            x = self.norm(x)

        if all_tokens:
            return x
        if self.global_pool:
            x = x[...,1:,:].mean(dim=1) # type: ignore
            return self.head(self.fc_norm(x)) 
        x = x[:,0] # type:ignore ## class token
        if cls_token: 
            return x
        return self.head(x)
    
class SPoT_MAE(nn.Module):
    
    def __init__(
        self, embed_dim, heads, depth, n_features=256,
        decoder_embed_dim=512, decoder_heads=16, decoder_depth=8,
        ksize=16, n_sigma=1.0, in_channels=3, pre_norm=False, mae_drop=0.75,
        **kwargs
   ):
        super().__init__()
        assert embed_dim % 2 == 0
        assert depth % 2 == 0
        self.encoder = SPoT(
            embed_dim, heads, depth, n_features=n_features,
            ksize=ksize, n_sigma=n_sigma, in_channels=in_channels, 
            pre_norm=pre_norm, mae_drop=mae_drop, 
            **kwargs
        )
        self.decoder = SPoT(
            decoder_embed_dim, decoder_heads, decoder_depth, n_features=n_features, 
            ksize=ksize, n_sigma=n_sigma, in_channels=in_channels, 
            pre_norm=pre_norm, mae_drop=mae_drop, decoder=True,
            encoder_embed_dim=embed_dim, **kwargs            
        )
        self.mae_drop = mae_drop
        self.head = nn.Linear(decoder_embed_dim, ksize**2*in_channels)
        self._mask = None

    @property
    def mask(self):
        mask = self._mask
        self._mask = None
        return mask
    
    @mask.setter
    def mask(self, val):
        self._mask = val

    def _proc_out(self, x, neg_mask, pred=False):
        x = apply_mae_mask(x, neg_mask, 1-self.mae_drop)
        B, N, E = x.shape
        ksize = self.encoder.embedder.ksize
        ch = self.encoder.embedder.channels - 1
        if pred:
            return x.view(B,N,ksize,ksize,ch).permute(0,1,4,2,3)
        return x[...,:ch*ksize**2].view(B,N,ch,ksize,ksize)

    
    def forward(self, x, mae_encode=True, keep_mu=False, keep_mask=False):
        # Encode inputs
        xh, x, mask = self.encoder(x, mae_encode=mae_encode, keep_mu=keep_mu)
        if keep_mask:
            self.mask = mask
            
        if mae_encode:
            # Decode inputs
            xh = self.decoder(xh, x, mask)
            
            # Final projection, disregarding class token
            xh = self.head(xh)[:,1:]

            # Precompute negative mask without class tokens
            neg_mask = ~mask[:,1:]
            return (
                self._proc_out(xh, neg_mask, pred=True), 
                self._proc_out(x, neg_mask, pred=False)
            )
        # Return pure encoder predictions
        return xh