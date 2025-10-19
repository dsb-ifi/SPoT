import torch
import quix
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from quix import AbstractLogger
import os

from spot.optim import LARS
from spot.nn import SPoTClassifier
from torch.optim import Optimizer
from torch.distributed.optim.zero_redundancy_optimizer import ZeroRedundancyOptimizer

_architecture_cfg = {
    'T': {'depth':12, 'embed_dim': 192, 'heads': 3, 'dop_path':0.0},
    'S': {'depth':12, 'embed_dim': 384, 'heads': 6, 'dop_path':0.1},
    'M': {'depth':12, 'embed_dim': 512, 'heads': 8, 'dop_path':0.1},
    'B': {'depth':12, 'embed_dim': 768, 'heads':12, 'dop_path':0.2},
    'L': {'depth':24, 'embed_dim':1024, 'heads':16, 'dop_path':0.2},
    'H': {'depth':32, 'embed_dim':1280, 'heads':16, 'dop_path':0.2},
}

class SPoTModelConfig(quix.ModelConfig):
    '''SPoT ModelConfig

    Attributes
    ----------
    n_features : int
        The number of blobs for the model.
    qkv_bias : bool
        Use bias for QKV matrices.
    n_classes : int
        The number of classes to predict.
    init_n_sigma : bool
        Overwrite n_sigma from checkpoint with given n_sigma argument.
    n_sigma : float
        Number of standard deviations to include in the Gaussian points.
    logprior : float
        Log prior for controlling blob precision / variance.
    learnable_n_sigma : bool
        If true, n_sigma is a learnable parameter.
    learnable_logprior : bool
        If true, logprior is a learnable parameter.
    pretrain_pos : bool
        If true, train only positional embedding weights.
    dop_path : Optional[float]
        Drop path rate to randomly drop MHSA or FFN layer in the ViT. If None,
        default rate for the ViT architechture is used. Defaults to None.
    llrd : float
        Layer-wise learning rate decay. Defaults to 1.0 i.e. no decay.
    sampler : str
        Which sampler to use in RandomGaussianTokenExtractor
    linear_probing : bool
        Flag to do linear probing
    '''
    n_features:int = 256
    qkv_bias:bool = True
    n_classes:int = 1000
    init_n_sigma:bool = False
    n_sigma:float = 1.5
    logprior:float = 1.5
    learnable_n_sigma:bool = False
    learnable_logprior:bool = False
    pretrain_pos:bool = False
    dop_path:Optional[float] = None
    llrd:float=1.0
    sampler:str='uniform'
    linear_probing:bool=False


class nSigmaLogger(AbstractLogger):

    def __init__(self, *args, **kwargs):
        super().__init__(['n_sigma'])



class SPoTRunner(quix.Runner):

    optimizer_dict = { # Fix later
        'adamw': torch.optim.AdamW,
        'adam': torch.optim.Adam,
        'lars': LARS
    }

    @property
    def mod(self) -> SPoTModelConfig:
        return self.cfg.mod # type: ignore
    
    @property
    def opt(self) -> quix.OptimizerConfig: #SPoTOptimizerConfig:
        return self.cfg.opt # type: ignore
    
    def parse_model(self):
        capacity = self.mod.model[0]
        ksize = int(self.mod.model[1:])
        if capacity in _architecture_cfg:
            modeldict = _architecture_cfg[capacity]
        else:
            raise ValueError(f'No valid architecture found for {self.mod.model}')
        if self.mod.dop_path is not None:
            modeldict['dop_path'] = self.mod.dop_path

        modeldict['ksize'] = ksize
        modeldict['n_features'] = self.mod.n_features 
        modeldict['qkv_bias'] = self.mod.qkv_bias
        modeldict['n_classes'] = self.mod.n_classes
        modeldict['logprior'] = self.mod.logprior
        modeldict['n_sigma'] = self.mod.n_sigma
        modeldict['learnable_n_sigma'] = self.mod.learnable_n_sigma
        modeldict['learnable_logprior'] = self.mod.learnable_logprior
        modeldict['pretrain_pos'] = self.mod.pretrain_pos
        modeldict['sampler'] = self.mod.sampler
        model = SPoTClassifier(**modeldict)

        if self.mod.pretrain_pos:
            for name, param in model.named_parameters():
                if name != 'embedder.emb.weight':
                    param.requires_grad = False

        if self.mod.linear_probing:
            # Add batch norm to model head
            model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
            # Freeze all layers but the head
            for name, param in model.named_parameters():
                param.requires_grad = False
            for name, param in model.head.named_parameters():
                param.requires_grad = True

        return model


    def parse_logger(self):
        rank = local_rank = 0
        if self.distributed:
            if self.rank is not None:
                rank = self.rank
            if self.local_rank is not None:
                local_rank = self.local_rank
        loggers = [
            quix.ProgressLogger(), quix.DeltaTimeLogger(), quix.LossLogger(), 
            quix.AccuracyLogger(top_k=1), quix.AccuracyLogger(top_k=5), nSigmaLogger(),
            quix.LRLogger(), quix.GPULogger()
        ]
        custom_runid = (
            self.__class__.__name__ + '_' + self.mod.model
            if self.log.custom_runid is None else self.log.custom_runid
        )        
        return quix.LogCollator(
            custom_runid,
            self.savedir,
            rank,
            local_rank,
            loggers,
            stdout=self.log.stdout
        )
    
    def preprocess_checkpoint(self, model, checkpoint):
        if self.mod.init_n_sigma:
                checkpoint['model']['tokenizer._n_sigma'] = model.tokenizer._init_n_sigma(self.mod.n_sigma)

        return checkpoint

    def parse_checkpoint(self, model, optimizer, scheduler, scaler, model_ema) -> int:
        if self.distributed:
            model = model.module

        start_epoch = self.cfg.start_epoch

        if self.mod.resume:
            if not os.path.isfile(self.mod.resume):
                raise FileNotFoundError(f'Invalid checkpoint resume path {self.mod.resume}')
            checkpoint = torch.load(self.mod.resume, map_location='cpu')
            checkpoint = self.preprocess_checkpoint(model, checkpoint)

            model.load_state_dict(checkpoint['model'])
            model.to(dtype=torch.get_default_dtype())

            if not self.mod.onlyweights:
                if not self.cfg.test_only:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    if scheduler:
                        scheduler.load_state_dict(checkpoint['scheduler'])

                start_epoch = checkpoint['epoch'] + 1 # Increment by one from checkpoint

                if model_ema: # TODO: ?
                    model_ema.load_state_dict(checkpoint['model_ema'])

                if scaler:
                    scaler.load_state_dict(checkpoint['scaler'])

        # Initialize checkpoint directory
        os.makedirs(self.checkpointdir, exist_ok=True)
        return start_epoch


    def parse_param_groups(self, model):
        depth = len(model.blocks)

        use_pre_depth = False
        use_post_depth = False

        base_weight_decay = self.opt.weight_decay
        base_learning_rate = self.opt.lr
        layer_decay = self.mod.llrd

        params = {
            'pre_decay': [],
            'pre_nodecay': [],
            **{f'block_{i}_decay': [] for i in range(depth)},
            **{f'block_{i}_nodecay': [] for i in range(depth)},
            'post_decay': [],
            'post_nodecay': [],
        }
        
        post_names = ['head', 'norm']
        
        for name, param in model.named_parameters():
            nm = name.split('.')
            if nm[0] in post_names:
                if param.numel() in param.shape:
                    params['post_nodecay'].append(param)
                else:
                    params['post_decay'].append(param)
            elif nm[0] == 'blocks':
                cur_depth = nm[1]
                if param.numel() in param.shape:
                    params[f'block_{cur_depth}_nodecay'].append(param)
                else:
                    params[f'block_{cur_depth}_decay'].append(param)
            else:
                if param.numel() in param.shape:
                    params['pre_nodecay'].append(param)
                else:
                    params['pre_decay'].append(param)
        
        block_scaling = [layer_decay**i for i in range(use_pre_depth, depth)]
        post_exp = use_post_depth + use_pre_depth + depth
        post_scaling = 1 if not use_post_depth else layer_decay**post_exp
        
        scaling = {
            'pre_decay': 1.0,
            'pre_nodecay': 1.0,
            **{f'block_{i}_decay': s for i,s in enumerate(block_scaling)},
            **{f'block_{i}_nodecay': s for i,s in enumerate(block_scaling)},    
            'post_decay': post_scaling,
            'post_nodecay': post_scaling,
        }
        
        return [
            {
                'params': params[key], 
                'lr': base_learning_rate * scaling[key],
                'weight_decay': 0. if key.endswith('nodecay') else base_weight_decay,
            } for key in params
        ]


    @staticmethod
    def forward_fn(inputs, targets, model, loss_fn):
        outputs = model(*inputs)
        loss = loss_fn(outputs, *targets)
        n_sigma = model.module.tokenizer.n_sigma.item()
        return {'outputs':outputs, 'loss':loss, 'n_sigma':n_sigma}

    
if __name__ == '__main__':
    runcfg = quix.RunConfig.argparse(modcfg=SPoTModelConfig, optcfg=quix.OptimizerConfig)
    SPoTRunner(runcfg).run()
