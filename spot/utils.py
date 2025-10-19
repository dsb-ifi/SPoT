import torch
import torch.nn.functional as F
from torch import Tensor
from PIL import Image
from typing import Optional, Union
from torch.distributions import Gamma
import numpy as np
import os


sobol_engine = torch.quasirandom.SobolEngine(2, True)

def custom_grid_sampler_2d(img:Tensor, grid:Tensor, mode:str='bilinear') -> Tensor:
    '''Custom 2D Grid Sampler for batch-wise sampling.

    Parameters
    ----------
    img : Tensor
        Image tensor of shape [B, C, H, W].
    grid : Tensor
        Grid normalized to [0,1] of shape [B, N, H_out, W_out, 2].
    mode : str
        Interpolation mode, either `bilinear` or `nearest`.

    Returns
    -------
    Tensor
        The interpolated tensor.
    '''
    B, C, H, W = img.shape
    B, N, H_out, W_out, _ = grid.shape
    
    # Separate y and x coordinates
    y_coords, x_coords = grid[..., 0] * H, grid[..., 1] * W
    b = torch.arange(B, device=img.device).view(-1, 1, 1, 1).expand(-1, N, H_out, W_out)
    
    if mode == 'nearest':
        # Calculate and round coordinates
        x0, y0 = x_coords.round().clamp(0, W-1), y_coords.round().clamp(0, H-1)
        output = img[b, :, y0, x0]

    elif mode == 'bilinear':
        # Calculate the four corner coordinates for interpolation
        x0, x1 = x_coords.floor().long(), (x_coords + 1).floor().long()
        y0, y1 = y_coords.floor().long(), (y_coords + 1).floor().long()
        
        # Ensure the coordinates are within image dimensions
        x0, x1 = x0.clamp(0, W-1), x1.clamp(0, W-1)
        y0, y1 = y0.clamp(0, H-1), y1.clamp(0, H-1)
        
        # Extract values from image
        Ia = img[b, :, y0, x0]
        Ib = img[b, :, y1, x0]
        Ic = img[b, :, y0, x1]
        Id = img[b, :, y1, x1]
        
        # Calculate interpolation weights
        wa = (x1 - x_coords) * (y1 - y_coords)
        wb = (x1 - x_coords) * (y_coords - y0)
        wc = (x_coords - x0) * (y1 - y_coords)
        wd = (x_coords - x0) * (y_coords - y0)
        
        # Perform bilinear interpolation
        output = wa.unsqueeze(-1) * Ia + wb.unsqueeze(-1) * Ib + wc.unsqueeze(-1) * Ic + wd.unsqueeze(-1) * Id
    
    else:
        raise NotImplementedError(f'No such intermolation mode: {mode}')
    
    return output


# Helper functions for gaussian_mixture_sampler
def _gms_get_grid(ksize:int, device:torch.device=torch.device('cpu')) -> Tensor:
    ls = torch.linspace(0, 1, ksize, device=device)
    return torch.stack(torch.meshgrid(ls, ls, indexing='ij'), -1)


def _gms_sample_mus(
    batch_size:int, n_feat:int, sampler:str, device:torch.device, 
    ksize:int, imgsize:tuple[int, int] | None = None
) -> Tensor:
    mu_shape = (batch_size, n_feat, 1, 1, 2)
    if sampler == 'uniform':
        return torch.rand(*mu_shape, device=device)
    elif sampler == 'uniform_perturb':
        assert imgsize is not None, 'imgsize must be provided for uniform_perturb sampler'
        h, w = imgsize
        mus = _sample_centroids(imgsize, n_feat).permute(1,0).view(1, -1, 1, 1, 2)
        mus = _perturb_grid_centers_uniform(mus, ksize)
        mus = mus.expand(batch_size, -1, 1, 1, 2)
        return mus.to(device)
    elif sampler == 'gauss_perturb':
        assert imgsize is not None, 'imgsize must be provided for uniform_perturb sampler'
        h, w = imgsize
        mus = _sample_centroids(imgsize, n_feat).permute(1,0).view(1, -1, 1, 1, 2)
        mus = _perturb_grid_centers_gauss(mus, ksize)
        mus = mus.expand(batch_size, -1, 1, 1, 2)
        return mus.to(device)
    elif sampler == 'gauss':
        mus = torch.randn(*mu_shape, device=device) / torch.pi + 0.5
        return torch.clamp(mus, 0, 1)
    elif sampler == 'sobol':
        smus = []
        for b in range(batch_size):
            smu = sobol_engine.draw(n_feat).view(*mu_shape[1:])
            smus.append(smu)
        return torch.stack(smus,0).to(device=device)
    elif sampler == 'grid_uniform':
        assert imgsize is not None, 'imgsize must be provided for grid_uniform sampler'
        h, w = imgsize
        mus = _sample_centroids(imgsize, n_feat).permute(1,0).view(1, -1, 1, 1, 2)
        mus = mus.expand(batch_size, -1, 1, 1, 2)
        return mus.to(device)
    elif sampler == 'grid_center':
        assert imgsize is not None, 'imgsize must be provided for grid_center sampler'
        h, w = imgsize
        mus = _sample_centroids_gauss(imgsize, n_feat).permute(1,0).view(1, -1, 1, 1, 2)
        mus = mus.expand(batch_size, -1, 1, 1, 2)
        return mus.to(device)
    elif sampler == 'edge':
        p = 10.0
        mus = torch.rand(*mu_shape, device=device).mul(2).sub(1)
        mus = mus.sign() * (1-mus.abs().pow(p))
        mus = (mus + 1.0) / 2.0
        return mus.clamp(1e-2, 1.-1e-2)
    else:
        raise ValueError(f'Unknown sampler: {sampler}')


def _gms_sample_sigmas(logprior:Tensor, base_sigma:Union[float, Tensor], batch_size:int, n_feat:int) -> Tensor:
    # rsample samples from the device of logprior
    sigma_shape = (batch_size, n_feat, 1, 1, 2)
    prior = logprior.exp()
    gam = Gamma(prior, prior)
    return base_sigma / gam.rsample(sigma_shape) # type: ignore


def _gms_get_gaussian_mask(grid:Tensor, n_sigma:Union[float, Tensor]) -> Tensor:
    grid = 2*grid - 1
    prec = 1/n_sigma
    gaussian = torch.exp(-prec**2*(grid[...,0]**2 + grid[...,1]**2) / 2)[None, None, None]
    return gaussian.div(gaussian.max())


def _gms_scaled_grid_sampler(img:Tensor, grid:Tensor, mu:Tensor, sigma:Union[float,Tensor], n_sigma:Union[float, Tensor]) -> Tensor:
    prec = 1/n_sigma
    scale = prec * sigma
    grid = grid[None, None] * (2 * scale) - scale
    return custom_grid_sampler_2d(img, grid + mu).permute(0,1,4,2,3)


def _gms_get_gaussian_pos(grid:Tensor, mu:Tensor, sigma:Union[float,Tensor]) -> Tensor:
    if isinstance(sigma, float):
        sigma_y = sigma_x = sigma
    elif isinstance(sigma, Tensor):
        if sigma.ndim == 1:
            sigma_y = sigma_x = sigma
        else:
            sigma_y = sigma[...,0]
            sigma_x = sigma[...,1]
    else:
        raise ValueError(f'Got {type(sigma)=}, expected Tensor or float.')
    pos = torch.exp(
        - (
            (grid[None,None,:,:,0] - mu[...,0])**2 / (2*sigma_y**2) +  
            (grid[None,None,:,:,1] - mu[...,1])**2 / (2*sigma_x**2)
        )
    ).unsqueeze(2)
    return pos.div(pos.max()) # Inplace operation caused error here...

# Gaussian Mixture Sampler
def gaussian_mixture_sampler_2d(
    img:Tensor, n_feat:int=256, ksize:int=32, n_sigma:Union[float,Tensor]=1.5, 
    mu:Optional[Tensor]=None,
    grid:Optional[Tensor]=None,
    gaussian:Optional[Tensor]=None,
    add_pos:bool=True,
    sampler:str='sobol',
    logprior:Optional[Tensor]=None,
    apply_gaussian:bool=False,
) -> Tensor:
    '''Gaussian 2D mixture sampler.

    NOTE: The actual sigma of the kernel corresponds exactly with:
    `n_sigma * (ksize - 1) / 2`

    Parameters
    ----------
    img : Tensor
        Image tensor of shape [B, C, H, W].
    n_feat : int
        Number of mixture features to extract.
    ksize : int
        Desired feature size.
    n_sigma : float
        Number of standard deviations to include in the Gaussian points.
    mu : Tensor
        Positions of shape [B, N, 1, 1, 2]
    grid : Tensor
        Mesh of shape [ksize, ksize, 2] with bounds (0, 1)
    gaussian : Tensor
        Gaussian mask of shape [1, 1, 1, ksize, ksize]
    add_pos : bool
        Flag for adding positional embeddings.
    sampler : str
        Sampling with uniform or quasi-MC method (Sobol engine).
    logprior : Optional[Tensor]
        Log prior for controlling blob precision / variance.
    apply_gaussian : bool
        Whether to apply the Gaussian mask.

    Returns
    -------
    Tensor
        The interpolated tensor.
    '''
    batch_size, channels, height, width = img.shape
    prec = 1/n_sigma
    sigma = ksize / (prec*height + prec*width)
    if logprior is not None:
        sigma = _gms_sample_sigmas(logprior, sigma, batch_size, n_feat)
    device = img.device
    
    if grid is None:
        grid = _gms_get_grid(ksize, img.device)
    else:
        if grid.shape != (ksize, ksize, 2): 
            raise ValueError('Error in grid shape!')
        
    if mu is None:
        mu = _gms_sample_mus(batch_size, n_feat, sampler, device, ksize=ksize, imgsize=(height, width))
    else:
        if mu.shape != (batch_size, n_feat, 1, 1, 2):
            raise ValueError('Error in mu shape!')
    
    features = _gms_scaled_grid_sampler(img, grid, mu, sigma, n_sigma)

    if apply_gaussian:
        if gaussian is None:
            gaussian = _gms_get_gaussian_mask(grid, n_sigma)
        else:
            if gaussian.shape != (1, 1, 1, ksize, ksize):
                raise ValueError('Error in gaussian shape!')
        features.mul_(gaussian)
       
    if add_pos:
        pos = _gms_get_gaussian_pos(grid, mu, sigma)
        features = torch.cat([features, pos], 2)
        
    return features

def reshape_patches(feat, C=4, H=32, W=32):
    '''Formats patches to original size. 
    '''
    B, N, E = feat.shape
    return feat.view(B, N, C, H, W).permute(0,1,3,4,2)


def sample_mae_mask(feat, mae_drop=0.75):
    '''Samples a MAE mask for selecting blobs.
    
    Parameters
    ----------
    feat : Tensor
        Features of shape [B, N, E]
    mae_drop : float, optional
        Probability for dropping tokens.
    
    Returns
    -------
    Tensor
        Mask with tokens to keep of shape [B, N]
    '''
    B, N, E = feat.shape
    n_keep = int(round((1 - mae_drop) * N))
    mask = feat.new_zeros(B, N, dtype=bool)
    sampled_indices = torch.rand(B, N, device=feat.device).multinomial(n_keep).view(-1)
    b_indices = torch.arange(B, device=feat.device).view(B,1).expand(B,n_keep).reshape(-1)
    mask[b_indices, sampled_indices] = True
    return mask


def apply_mae_mask(feat, mask, mae_drop=0.75):
    '''Applies a MAE mask to the input feature tensor.
    
    Parameters
    ----------
    feat : Tensor
        Features of shape [B, N, E]
    mask : Tensor
        Boolean mask of shape [B, N] of which tokens to keep
    mae_drop : float, optional
        Probability for dropping tokens. 
    
    Returns
    -------
    Tensor
        Masked features of shape [B, n_keep, E].
    '''    
    B, N, E = feat.shape
    n_keep = int(round((1 - mae_drop) * N))
    return feat[mask].reshape(B, n_keep, E)


def _sample_centroids(shape, n, normalized=True, mode='round') -> torch.Tensor:
    shape = torch.tensor(shape).long()
    nd = len(shape)
    sorted_dims, unsort_dim_idxs = torch.sort(shape)
    sizes = shape.prod().double()
    steps = torch.ones(nd) * (sizes / n)**(1/nd)

    if (sorted_dims < steps).any():
        for d in range(nd):
            steps[d] = sorted_dims[d]
            sizes = torch.prod(sorted_dims[d+1:]).float()
            steps[d+1:] = (sizes / n)**(1/(nd-d-1))
            if (sorted_dims >= steps).all():
                break

    starts = steps.div(2, rounding_mode='trunc').int()

    if mode == 'round':
        steps = torch.round(steps).int()
    elif mode == 'floor':
        steps = torch.floor(steps).int()
    elif mode == 'ceil':
        steps = torch.ceil(steps).int()
    else:
        raise ValueError(f'Invalid mode: {mode}.')

    slices = tuple(
        slice(starts[i].item(), None, steps[i].item()) 
        for i in unsort_dim_idxs
    )
    grid = torch.tensor(np.mgrid[tuple(slice(None, a.item(), None) for a in shape)])

    centroids = torch.stack([
        grid[i][slices].ravel()
        for i in range(nd)
    ])
    if not normalized:
        return centroids
    return torch.einsum('d...,d->d...', centroids, shape.reciprocal())


def _sample_centroids_gauss(shape, n_feat, s=0.2, normalized=True, mode='round'):
    coords = _sample_centroids(shape, n_feat, normalized, mode)
    delta = coords - 0.5
    w = 1-torch.exp(-(delta / s) ** 2)
    return 0.5 + delta * w

def _get_mask(_name, path, extension='png', invert=False):
    filename = f"{_name}.{extension}"
    full_path = os.path.join(path, filename)
    mask = np.array(Image.open(full_path))
    if invert:
        mask = 1. - torch.tensor(mask / 255, dtype=torch.float32)
    else:
        mask = torch.tensor(mask / 255, dtype=torch.float32)
    return mask

def _sample_mus_from_mask(_name, n_samples, path, imgsize=224, eps=1e-2, extension='png', invert=False):    
    mask = _get_mask(_name, path, extension, invert)
    cdf = mask.div(mask.sum()).view(-1).cumsum(-1)
    support = torch.stack(
        torch.meshgrid(
            [torch.linspace(0,1,imgsize),]*2,
            indexing='ij'
        ), -1
    ).view(-1,2)    
    rand = torch.rand(n_samples).clamp(max=cdf[-1].item()-1e-12)
    mus = support[torch.searchsorted(cdf, rand)]
    mus = (mus + torch.rand_like(mus) / imgsize).clamp(eps,1-eps)
    return mus.mT

def _sample_square_spiral(n_feat, num_traversals=8, normalized=True):
    if not normalized:
        raise ValueError('Spiral sampling only supports normalized coordinates.')
    r = torch.linspace(0, 1, n_feat)
    th = torch.linspace(0, num_traversals * torch.pi, n_feat)
    u = torch.cos(th)
    v = torch.sin(th)
    m = torch.max(u.abs(), v.abs())
    x = r * u / m
    y = r * v / m
    return torch.stack([x, y], dim=0)

def _perturb_grid_centers_uniform(mu:Tensor, ksize:int) -> Tensor:
    zone_width = 1.0 / ksize
    offset = (torch.rand_like(mu) - 0.5) * zone_width
    return mu + offset

def _perturb_grid_centers_gauss(mu:Tensor, ksize:int, scale=1.0) -> Tensor:
    zone_width = 1.0 / ksize
    std = scale * (zone_width / 2.0)
    offset = torch.randn_like(mu) * std
    return mu + offset

def reconstruct_image_from_predictions(
    pred:Tensor, mu:Tensor, mask:Tensor, grid:Tensor, 
    ksize:int, n_sigma:float, mae_drop:float, 
    n_features:int, img_shape:tuple[int,int,int,int],
    _return_combine:bool=False
) -> Tensor:
    '''Reconstructs an image using predicted patches.

    Parameters
    ----------
    pred : Tensor
        Tensor with patch predictions.
    mu : Tensor
        Positions of shape [B, N, 1, 1, 2]
    mask : Tensor
        Boolean mask of shape [B, N] of which tokens to keep.
    grid : Tensor
        Mesh of shape [ksize, ksize, 2] with bounds (0, 1)
    ksize : int
        Desired feature size.
    n_sigma : float
        Number of standard deviations to include in the Gaussian points.
    mae_drop : float, optional
        Probability for dropping tokens. 
    n_feat : int
        Number of mixture features to extract.
    img_shape : tuple[int,int,int,int]
        Tuple with image size; [B, C, H, W].

    Returns
    -------
    Tensor
        Reconstructed image of shape [B, C, H, W].
    '''
    B, C, H, W = img_shape
    sigma = ksize * n_sigma / (H + W)
    scale = sigma / n_sigma
    negmask = ~mask[:,1:]
    n_keep = int(round(mae_drop * n_features))
    mukeep = mu[:,:,0,0][negmask].view(mu.shape[0], n_keep, 1, 1, 2)
    mugrid = (grid * (2 * scale) - scale)[None,None] + mukeep
    B, N, H_out, W_out, _ = mugrid.shape

    # Separate y and x coordinates
    y_coords, x_coords = mugrid[..., 0] * H, mugrid[..., 1] * W

    # Calculate the four corner coordinates
    x0, x1 = x_coords.floor().long(), (x_coords + 1).floor().long()
    y0, y1 = y_coords.floor().long(), (y_coords + 1).floor().long()
    b = torch.arange(B, device=grid.device).view(-1, 1, 1, 1).expand(-1, N, H_out, W_out)

    # Remove any coordinates outside image
    yxmask = (x0 >= 0) & (x1 <= (W-1)) & (y0 >= 0) & (y1 <= (W-1))

    # Mask all relevant tensors
    feat = pred.permute(0,1,3,4,2)[yxmask]
    b, y_coords, x_coords = b[yxmask], y_coords[yxmask], x_coords[yxmask]
    x0, x1, y0, y1 = x0[yxmask], x1[yxmask], y0[yxmask], y1[yxmask]

    # Compute interpolation weights
    wa = (x1 - x_coords) * (y1 - y_coords) # y0x0
    wb = (x1 - x_coords) * (y_coords - y0) # y1x0
    wc = (x_coords - x0) * (y1 - y_coords) # y0x1
    wd = (x_coords - x0) * (y_coords - y0) # y1x1

    # Construct output tensors
    out = pred.new_zeros(4,B*H*W, C)
    den = pred.new_zeros(4,B*H*W, 1)

    # Fill denominator
    den[0].scatter_add_(0, (b * H * W + y0*W + x0).view(-1,1), wa.view(-1,1))
    den[1].scatter_add_(0, (b * H * W + y1*W + x0).view(-1,1), wb.view(-1,1))
    den[2].scatter_add_(0, (b * H * W + y0*W + x1).view(-1,1), wc.view(-1,1))
    den[3].scatter_add_(0, (b * H * W + y1*W + x1).view(-1,1), wd.view(-1,1))

    # Fill output
    out[0].scatter_add_(0, (b * H * W + y0*W + x0).view(-1,1).expand(-1,C), feat * wa.view(-1,1))
    out[1].scatter_add_(0, (b * H * W + y1*W + x0).view(-1,1).expand(-1,C), feat * wb.view(-1,1))
    out[2].scatter_add_(0, (b * H * W + y0*W + x1).view(-1,1).expand(-1,C), feat * wc.view(-1,1))
    out[3].scatter_add_(0, (b * H * W + y1*W + x1).view(-1,1).expand(-1,C), feat * wd.view(-1,1))

    # Sum over interpolations
    out, den = out.sum(0), den.sum(0)

    if _return_combine:
        return torch.cat([out, den], -1)

    # Mask all unfilled features
    denmask = den > 0    
    out[denmask[:,0]] = out[denmask[:,0]] / den[denmask].view(-1,1).expand(-1,C)        

    # Return reconstruction
    return out.view(B,H,W,C).permute(0,3,1,2)

def reconstruct_image_from_actual(
    real:Tensor, mu:Tensor, mask:Tensor, grid:Tensor, gaussian:Tensor,
    ksize:int, n_sigma:float, mae_drop:float, 
    n_features:int, img_shape:tuple[int,int,int,int],
    _return_combine:bool=False
) -> Tensor:
    '''Reconstructs an image using actual patches.

    Parameters
    ----------
    real : Tensor
        Tensor with actual patches (w.o. positional embeddings).
    mu : Tensor
        Positions of shape [B, N, 1, 1, 2]
    mask : Tensor
        Boolean mask of shape [B, N] of which tokens to keep.
    grid : Tensor
        Mesh of shape [ksize, ksize, 2] with bounds (0, 1)
    gaussian : Tensor
        Gaussian mask of shape [1, 1, 1, ksize, ksize]
    ksize : int
        Desired feature size.
    n_sigma : float
        Number of standard deviations to include in the Gaussian points.
    mae_drop : float, optional
        Probability for dropping tokens. 
    n_feat : int
        Number of mixture features to extract.
    img_shape : tuple[int,int,int,int]
        Tuple with image size; [B, C, H, W].

    Returns
    -------
    Tensor
        Reconstructed image of shape [B, C, H, W].
    '''
    B, C, H, W = img_shape
    sigma = ksize * n_sigma / (H + W)
    scale = sigma / n_sigma
    posmask = mask[:,1:]
    n_keep = int(round((1-mae_drop) * n_features))
    mukeep = mu[:,:,0,0][posmask].view(mu.shape[0], n_keep, 1, 1, 2)
    mugrid = (grid * (2 * scale) - scale)[None,None] + mukeep
    real = real[posmask].view(B, n_keep, C, ksize, ksize)
    gaussian = gaussian[:,:,0].expand(B, n_keep, ksize, ksize)
    B, N, H_out, W_out, _ = mugrid.shape
    print(mugrid.shape, "mugrid")

    # Separate y and x coordinates
    y_coords, x_coords = mugrid[..., 0] * H, mugrid[..., 1] * W

    # Calculate the four corner coordinates
    x0, x1 = x_coords.floor().long(), (x_coords + 1).floor().long()
    y0, y1 = y_coords.floor().long(), (y_coords + 1).floor().long()
    b = torch.arange(B, device=grid.device).view(-1, 1, 1, 1).expand(-1, N, H_out, W_out)

    # Remove any coordinates outside image
    yxmask = (x0 >= 0) & (x1 <= (W-1)) & (y0 >= 0) & (y1 <= (W-1))

    # Mask all relevant tensors
    feat, gaussian = real.permute(0,1,3,4,2)[yxmask], gaussian[yxmask]
    b, y_coords, x_coords = b[yxmask], y_coords[yxmask], x_coords[yxmask]
    x0, x1, y0, y1 = x0[yxmask], x1[yxmask], y0[yxmask], y1[yxmask]

    # Compute interpolation weights
    wa = (x1 - x_coords) * (y1 - y_coords) * gaussian # y0x0
    wb = (x1 - x_coords) * (y_coords - y0) * gaussian # y1x0
    wc = (x_coords - x0) * (y1 - y_coords) * gaussian # y0x1
    wd = (x_coords - x0) * (y_coords - y0) * gaussian # y1x1

    # Construct output tensors
    out = real.new_zeros(4,B*H*W, C)
    den = real.new_zeros(4,B*H*W, 1)

    # Fill denominator
    den[0].scatter_add_(0, (b * H * W + y0*W + x0).view(-1,1), wa.view(-1,1))
    den[1].scatter_add_(0, (b * H * W + y1*W + x0).view(-1,1), wb.view(-1,1))
    den[2].scatter_add_(0, (b * H * W + y0*W + x1).view(-1,1), wc.view(-1,1))
    den[3].scatter_add_(0, (b * H * W + y1*W + x1).view(-1,1), wd.view(-1,1))

    # Fill output
    out[0].scatter_add_(0, (b * H * W + y0*W + x0).view(-1,1).expand(-1,C), feat * wa.view(-1,1))
    out[1].scatter_add_(0, (b * H * W + y1*W + x0).view(-1,1).expand(-1,C), feat * wb.view(-1,1))
    out[2].scatter_add_(0, (b * H * W + y0*W + x1).view(-1,1).expand(-1,C), feat * wc.view(-1,1))
    out[3].scatter_add_(0, (b * H * W + y1*W + x1).view(-1,1).expand(-1,C), feat * wd.view(-1,1))

    # Sum over interpolations
    out, den = out.sum(0), den.sum(0)

    if _return_combine:
        return torch.cat([out, den], -1)

    # Mask all unfilled features
    denmask = den > 0    
    out[denmask[:,0]] = out[denmask[:,0]] / den[denmask].view(-1,1).expand(-1,C)        

    # Return reconstruction
    return out.view(B,H,W,C).permute(0,3,1,2)


def reconstruct_image_full(
    pred:Tensor, real:Tensor, mu:Tensor, mask:Tensor, grid:Tensor, gaussian:Tensor,
    ksize:int, n_sigma:float, mae_drop:float, 
    n_features:int, img_shape:tuple[int,int,int,int],
) -> Tensor:
    '''Reconstructs an image using both predictions and actual patches.

    Parameters
    ----------
    pred : Tensor
        Tensor with predicted patches.
    real : Tensor
        Tensor with actual patches (w.o. positional embeddings).
    mu : Tensor
        Positions of shape [B, N, 1, 1, 2]
    mask : Tensor
        Boolean mask of shape [B, N] of which tokens to keep.
    grid : Tensor
        Mesh of shape [ksize, ksize, 2] with bounds (0, 1)
    gaussian : Tensor
        Gaussian mask of shape [1, 1, 1, ksize, ksize]
    ksize : int
        Desired feature size.
    n_sigma : float
        Number of standard deviations to include in the Gaussian points.
    mae_drop : float, optional
        Probability for dropping tokens. 
    n_feat : int
        Number of mixture features to extract.
    img_shape : tuple[int,int,int,int]
        Tuple with image size; [B, C, H, W].

    Returns
    -------
    Tensor
        Reconstructed image of shape [B, C, H, W].
    '''
    B,C,H,W = img_shape
    recpred = reconstruct_image_from_predictions(
        pred, mu, mask, grid, ksize, n_sigma, mae_drop,
        n_features, img_shape, True
    )
    recreal = reconstruct_image_from_actual(
        real, mu, mask, grid, gaussian, ksize, n_sigma, mae_drop,
        n_features, img_shape, True
    )
    rec, den = recpred[:,:3] + recreal[:,:3], recpred[:,3:] + recreal[:,3:]
    denmask = den > 0
    rec[denmask[:,0]] = rec[denmask[:,0]] / den[denmask].view(-1,1).expand(-1,C)
    return rec.view(B,H,W,C).permute(0,3,1,2)