import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import os
import sys
import itertools


from spot.utils import (
    _sample_centroids, _sample_centroids_gauss, _sample_mus_from_mask,
    sobol_engine
)
from spot.cadam import AdamW as CAdamW

def _clear_agnostic(wait: bool = True):
    try:
        ip = get_ipython()  # type: ignore[name-defined]
        if ip is not None:
            from IPython.display import clear_output
            clear_output(wait=wait)
            return
    except Exception:
        pass

    if os.name == "nt":
        os.system("cls")
    else:
        sys.stdout.write("\033[H\033[J")
        sys.stdout.flush()

class OracleContext:

    _opt_dict = dict(
        adamw = torch.optim.AdamW,
        cadamw = CAdamW,
        sgd = torch.optim.SGD,
    )

    _valid_priors = [
        'uniform',
        'gauss',
        'grid_uniform',
        'grid_center',
        'sobol',
        'salient',
        'inv_salient',
        'edge'
    ]

    def __init__(
        self, model, n_feats, prior, optimizer, device, 
        steps=5, lr=3e-3, salient_imgs_path=None, loss_fn=nn.CrossEntropyLoss(), 
        compute_acc=True, optimize_cleaning=False, stdout=True, num_iterations=None,
        **opt_kwargs,
    ):
        assert optimizer in self._opt_dict
        assert prior in self._valid_priors
        if prior == 'salient' or prior=='inv_salient':
            msg = "Using salient prior requires a path to salient images."
            assert salient_imgs_path is not None, msg
        self.model = model
        self.n_feats = n_feats
        self.prior = prior
        self.optimizer = optimizer
        self.device = device
        self.steps = steps
        self.lr = lr
        self.path = salient_imgs_path
        self.loss_fn = loss_fn
        self.compute_acc = compute_acc
        self.optimize_cleaning = optimize_cleaning
        self.stdout = stdout
        self.opt_kwargs = opt_kwargs

        self.mus = []
        self.acs = []
        self.it = 0
        self.N = num_iterations

    def create_optimizer(self, mu):
        return self._opt_dict[self.optimizer](
            [mu], lr=self.lr, **self.opt_kwargs
        )

    def sample_prior_mus(self, shape, _name=None):
        device = self.device
        B, _, H, W = shape
        match self.prior:
            case 'uniform':
                mu = torch.rand(B, self.n_feats, 1, 1, 2).to(device)
            case 'gauss':
                mus = torch.randn(B, self.n_feats, 1, 1, 2, device=device) / torch.pi + 0.5
                return torch.clamp(mus, 0, 1)
            case 'grid_uniform':
                mu = _sample_centroids((H, W), self.n_feats)
                mu = mu.permute(1,0).view(1, -1, 1, 1, 2)
                mu = mu.expand(B, -1, 1, 1, 2).clone().to(device)                
            case 'grid_center':
                mu = _sample_centroids_gauss((H, W), self.n_feats, 0.3)
                mu = mu.permute(1,0).view(1, -1, 1, 1, 2)
                mu = mu.expand(B, -1, 1, 1, 2).clone().to(device)                
            case 'sobol':
                smus = []
                for b in range(B):
                    smus.append(sobol_engine.draw(self.n_feats).view(self.n_feats, 1, 1, 2))
                mu = torch.stack(smus,0).to(device)                
            case 'salient':
                mu = []
                assert _name is not None, "salient sampling requires a name for the sampling."
                for _n in _name:
                    mu += [_sample_mus_from_mask(_n, self.n_feats, self.path).mT.to(device)]
                mu = torch.stack(mu).view(B, self.n_feats, 1, 1, 2)
            case 'inv_salient':
                mu = []
                assert _name is not None, "salient sampling requires a name for the sampling."
                for _n in _name:
                    mu += [_sample_mus_from_mask(_n, self.n_feats, self.path, invert=True).mT.to(device)]
                mu = torch.stack(mu).view(B, self.n_feats, 1, 1, 2)
            case 'edge':
                p = 10.0
                mu = torch.rand(B, self.n_feats, 1, 1, 2).mul(2).sub(1).to(device)
                mu = mu.sign() * (1-mu.abs().pow(p))
                mu = (mu + 1.0) / 2.0
                mu = mu.clamp(1e-2, 1.-1e-2)
        return mu

    def __call__(self, imgs, labs, _name=None):
        '''Performs the oracle context optimization.
        
        Parameters
        ----------
        imgs : torch.Tensor
            A tensor of shape (B, C, H, W) containing the input images.
        labs : torch.Tensor
            A tensor of shape (B,) containing the labels for the images.
        _name : list of str, optional
            A list of names corresponding to the images, used for salient sampling.
            If not provided, the prior will be sampled uniformly.
            This is only used if the prior is 'salient'.
        Returns
        -------
        None
            This method performs the optimization in place and updates the mus 
            attribute, which is a list of tensors containing the points discovered
            during the oracle context optimization. The mus are stored in the self.mus 
            attribute as a list of tensors.

            NOTE: It also prints the progress / accuracy if stdout is set to True.
        '''
        device = self.device
        B, C, H, W = imgs.shape
        assert B == labs.shape[0], "Batch size of images and labels must match."
        assert self.n_feats <= H * W, "Number of features must be less than or equal to the number of pixels."

        imgs, labs = imgs.to(device), labs.to(device)
        mu = self.sample_prior_mus((B, C, H, W), _name=_name)
        cmus = [mu.clone().cpu().detach()]
        mu.requires_grad = True
        opt = self.create_optimizer(mu)

        if self.compute_acc and len(self.acs) > 0:
            aggr_acc = torch.cat(self.acs).float().mean().item()
        else:
            aggr_acc = 0.0

        for step in range(self.steps):
            opt.zero_grad(set_to_none=True)
            out = self.model(imgs, mu=mu)
            loss = self.loss_fn(out, labs)
            loss.backward()
            opt.step()
            cmus += [mu.clone().cpu().detach()]
            stdout = f"It: {self.it:8d}"
            if self.N is not None:
                stdout += f'/ {self.N:8d}'
            stdout += f" -- Step {step:6d} -- Loss: {loss.item():2.6f}"
            if self.compute_acc:
                acc = (out.argmax(1) == labs).float().mean().item()
                stdout += f" -- Acc: {acc:.4f} -- Aggr.Acc: {aggr_acc:.4f}"
            if self.stdout:
                _clear_agnostic(wait=True)
                print(stdout)
        
        if self.steps > 0:
            del out
            del loss
            if self.optimize_cleaning:
                gc.collect()
                gc.collect()
                torch.cuda.empty_cache()

        if self.compute_acc:
            with torch.no_grad():
                out = self.model(imgs, mu=mu)
                acc = (out.argmax(-1) == labs)
                loss = self.loss_fn(out, labs)

            self.acs.append(acc.cpu())
            aggr_acc = torch.cat(self.acs).float().mean().item()
            stdout = f"It: {self.it:8d}"
            if self.N is not None:
                stdout += f'/ {self.N:8d}'
            if self.steps > 0:
                stdout += f" -- Step {step+1:6d} -- Loss: {loss.item():2.6f}"
            acc = (out.argmax(1) == labs).float().mean().item()
            stdout += f" -- Acc: {acc:.4f} -- Aggr.Acc: {aggr_acc:.4f}"
            if self.stdout:
                _clear_agnostic(wait=True)
                print(stdout)

        self.it += 1
        self.mus.append(torch.stack(cmus, 1))

    def collect_mus(self):
        '''Collects the mus from the oracle context.
        
        Returns
        -------
        mus : torch.Tensor
            A tensor of shape (n_mus, steps+1, n_features, 2) containing the mus.
        '''
        if len(self.mus) == 0:
            return torch.tensor([])
        return torch.cat(self.mus, 0)[...,0,0,:]
    
    def collect_snapped_mus(self, height=224, width=224, n_features=196):
        '''Collects the mus and snaps them to a grid of centroids.

        This provides a way to measure how discovered features perform
        when snapped to a grid of centroids, as opposed to the subpixel
        locations they were discovered at.
        
        Parameters
        ----------
        height : int
            Height of the grid.
        width : int
            Width of the grid.
        n_features : int
            Number of features to sample.
        Returns
        -------
        grid_points : torch.Tensor
            A tensor of shape (n_mus, 1, 1, n_features, 2) containing the snapped mus.
        '''
        mus = self.collect_mus()
        centers = _sample_centroids((height, width), n_features).mT
        l2dist = (mus[:,-1].unsqueeze(-2) - centers.unsqueeze(-3)).pow(2).sum(-1).sqrt()
        grid_points = centers[l2dist.argmin(-1)][...,None,None,:]
        return grid_points

    def serialize_mus(self, path):
        '''Serializes the mus to a file.

        Parameters
        ----------
        path : str
            Path to the file where the mus will be saved.
        '''
        mus = self.collect_mus()
        torch.save(mus, path)

    def load_mus(self, path):
        '''Loads the mus from a file.

        Parameters
        ----------
        path : str
            Path to the file where the mus are saved.
        '''
        assert self.N is not None, "Cannot load mus without a defined number of iterations."
        mus = torch.load(path)
        self.n_feats = mus.shape[-2]
        self.steps = mus.shape[1] - 1
        mus = mus.view(-1, self.steps + 1, self.n_feats, 1, 1, 2)
        self.it = self.N
        self.mus = mus.chunk(self.N, 0)


# class SPoT_SSL(OracleContext):
#     def __init__(self, tau_divisor=64, chunks=12, **kwargs):
#         super().__init__(**kwargs)
#         self.tau = torch.ones(1)/tau_divisor
#         self.chunks = chunks
#     def ssl_loss(self, chunk1, chunk2):
#         return (2 - 2* F.cosine_similarity(chunk1, chunk2)).mean()

#     def __call__(self, imgs, labs, _name=None):
#         '''Performs the oracle context optimization.
        
#         Parameters
#         ----------
#         imgs : torch.Tensor
#             A tensor of shape (B, C, H, W) containing the input images.
#         labs : torch.Tensor
#             A tensor of shape (B,) containing the labels for the images.
#         _name : list of str, optional
#             A list of names corresponding to the images, used for salient sampling.
#             If not provided, the prior will be sampled uniformly.
#             This is only used if the prior is 'salient'.
#         Returns
#         -------
#         None
#             This method performs the optimization in place and updates the mus 
#             attribute, which is a list of tensors containing the points discovered
#             during the oracle context optimization. The mus are stored in the self.mus 
#             attribute as a list of tensors.

#             NOTE: It also prints the progress / accuracy if stdout is set to True.
#         '''
#         device = self.device
#         self.tau = self.tau.to(device)
#         B, C, H, W = imgs.shape
#         assert B == labs.shape[0], "Batch size of images and labels must match."
#         assert self.n_feats <= H * W, "Number of features must be less than or equal to the number of pixels."

#         imgs, labs = imgs.to(device), labs.to(device)
#         mu = self.sample_prior_mus((B, C, H, W), _name=_name)
#         cmus = [mu.clone().cpu().detach()]
#         mu.requires_grad = True
#         opt = self.create_optimizer(mu)

#         if self.compute_acc and len(self.acs) > 0:
#             aggr_acc = torch.cat(self.acs).float().mean().item()
#         else:
#             aggr_acc = 0.0

#         for step in range(self.steps):
#             opt.zero_grad(set_to_none=True)
#             outs = []
#             for l in range(self.chunks):
#                 out = self.model(imgs, mu=mu + torch.randn_like(mu)*F.softplus(self.tau), cls_token=True)
#                 outs.append(out)

#             loss = sum(                                   # fresh tensor each step
#                 self.ssl_loss(outs[i], outs[j])
#                 for i, j in itertools.combinations(range(self.chunks), 2)
#             )

#             loss.backward() # type: ignore[call-arg]
#             opt.step()
#             cmus += [mu.clone().cpu().detach()]
#             stdout = f"It: {self.it:8d}"
#             if self.N is not None:
#                 stdout += f'/ {self.N:8d}'
#             stdout += f" -- Step {step:6d} -- Loss: {loss.item():2.6f}" # type: ignore[call-arg]
#             if self.compute_acc:
#                 acc = (out.argmax(1) == labs).float().mean().item()
#                 stdout += f" -- Acc: {acc:.4f} -- Aggr.Acc: {aggr_acc:.4f}"
#             if self.stdout:
#                 _clear_agnostic(wait=True)
#                 print(stdout)
        
#         if self.steps > 0:
#             del out
#             del loss
#             if self.optimize_cleaning:
#                 gc.collect()
#                 gc.collect()
#                 torch.cuda.empty_cache()

#         if self.compute_acc:
#             with torch.no_grad():
#                 out = self.model(imgs, mu=mu)
#                 acc = (out.argmax(-1) == labs)
#                 loss = self.loss_fn(out, labs)

#             self.acs.append(acc.cpu())
#             aggr_acc = torch.cat(self.acs).float().mean().item()
#             stdout = f"It: {self.it:8d}"
#             if self.N is not None:
#                 stdout += f'/ {self.N:8d}'
#             if self.steps > 0:
#                 stdout += f" -- Step {step+1:6d} -- Loss: {loss.item():2.6f}"
#             acc = (out.argmax(1) == labs).float().mean().item()
#             stdout += f" -- Acc: {acc:.4f} -- Aggr.Acc: {aggr_acc:.4f}"
#             if self.stdout:
#                 _clear_agnostic(wait=True)
#                 print(stdout)

#         self.it += 1
#         self.mus.append(torch.stack(cmus, 1))