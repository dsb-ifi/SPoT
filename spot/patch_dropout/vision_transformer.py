from timm.models.vision_transformer import VisionTransformer 
from timm.models._manipulate import checkpoint_seq
import numpy as np
import torch
from torch import nn
from .patch_dropout import PatchDropout

class PatchDropoutVisionTransformer(VisionTransformer):
    """
    Vision Transformer extended with PatchDropout for token-level dropout.

    Inherits all configurable hyperparameters from timm’s VisionTransformer,
    adding two customization arguments:

      patch_drop_rate : float
        Probability of dropping each non-prefix patch token. If zero,
        no dropout is applied (Identity).
      dropout_strategy : {'random', 'object-centric', 
                          'background-centric', 'grid'}
        Strategy passed through to the internal PatchDropout module.

    Example
    -------
    >>> from timm.models import load_model_config_from_hf, build_model_with_cfg
    >>> cfg, variant, args = load_model_config_from_hf(
    ...     'timm/vit_base_patch16_224.augreg2_in21k_ft_in1k'
    ... )
    >>> model = build_model_with_cfg(
    ...     PatchDropoutVisionTransformer,
    ...     variant=variant,
    ...     patch_drop_rate=0.8,
    ...     dropout_strategy='object-centric',
    ...     pretrained_cfg=cfg,
    ...     pretrained=True
    ... )
    """
    def __init__(self, 
                 patch_drop_rate: float = 0.,
                 dropout_strategy: str = 'random',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                prob=patch_drop_rate,
                dropout_strategy=dropout_strategy,
                num_prefix_tokens=1,
            )
        else:
            self.patch_drop = nn.Identity()


    def forward_features(self, x: torch.Tensor, mask: np.array = None) -> torch.Tensor:
        """
        Compute transformer features with optional patch dropout.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (batch_size, 3, height, width).
        mask : np.array or None, optional
            Patch-level mask for dropout strategies that require it.
            Shape must be (batch_size, n_tokens, n_tokens). Default None.

        Returns
        -------
        torch.Tensor
            Sequence of hidden features after the transformer blocks,
            of shape (batch_size, num_tokens+1, embed_dim).
        """
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x, mask=mask)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor, mask : np.array = None, return_features: bool = False) -> torch.Tensor:
        features = self.forward_features(x, mask=mask)
        x = self.forward_head(features)
        if return_features:
            return x, features
        return x