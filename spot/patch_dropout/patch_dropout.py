from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class PatchDropout(nn.Module):
    """
    Drop a subset of patch tokens according to various strategies during
    Vision Transformer forward passes.

    This module implements stochastic token dropout as described in
    https://arxiv.org/abs/2212.00794 (“Token-to-Token Dropout”)
    and https://arxiv.org/pdf/2208.07220 (“PatchDropout”), with support for:
      - random: uniform random selection of tokens,
      - object-centric: sampling tokens concentrated on a provided binary mask,
      - background-centric: sampling tokens complementary to the mask,
      - grid: (placeholder) systematic grid-based selection.

    Parameters
    ----------
    prob : float, optional
        Dropout probability in [0, 1).  A value of 0 disables dropout
        (identity pass-through). Default is 0.5.
    num_prefix_tokens : int, optional
        Number of tokens at the start of each sequence to always keep
        (e.g., [CLS] or other learned prefix tokens). Default is 1.
    ordered : bool, optional
        If True, the kept token indices are sorted to preserve original
        spatial order (useful for visualization). Default is False.
    return_indices : bool, optional
        If True, the module returns a tuple (dropped_tokens, keep_indices)
        instead of just the dropped_tokens. Default is False.
    dropout_strategy : {'random', 'object-centric', 
                        'background-centric', 'grid'}, optional
        Strategy controlling which tokens to drop.  Default is 'random'.

    Methods
    -------
    forward(x, mask=None)
        Apply token dropout to input embeddings.

    Notes
    -----
    - Expects `mask` to be a NumPy array (or torch tensor) of shape
      (batch_size, n_tokens, n_tokens), with values in [0, 255].
    - For object-centric and background-centric modes, mask is normalized
      to [0,1] and used as sampling weights.
    - Grid mode is not yet implemented.
    """
    return_indices: torch.jit.Final[bool]

    def __init__(
            self,
            prob: float = 0.5,
            num_prefix_tokens: int = 1,
            ordered: bool = False,
            return_indices: bool = False,
            dropout_strategy: str = 'random',
    ):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.num_prefix_tokens = num_prefix_tokens  # exclude CLS token (or other prefix tokens)
        self.ordered = ordered
        self.return_indices = return_indices
        self.dropout_strategy = dropout_strategy
    
    def get_object_indices(self, x, mask, batch_size, num_keep):
        mask = torch.tensor(mask / 255, dtype=x.dtype, device=x.device).view(batch_size, -1)
        keep_indices = mask.multinomial(num_keep, replacement=False)
        return keep_indices

    def get_background_indices(self, x, mask, batch_size, num_keep):
        mask = torch.tensor(mask / 255, dtype=x.dtype, device=x.device).view(batch_size, -1)
        keep_indices = (1-mask).multinomial(num_keep, replacement=False)
        return keep_indices
    
    def get_grid_indices(self, x, mask, batch_size, num_keep):
        raise NotImplementedError("Grid dropout strategy is not implemented yet.")

    def forward(self, x, mask=None) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Perform patch token dropout on the input sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input embedding tokens of shape
            (batch_size, num_prefix_tokens + num_tokens, embedding_dim).
        mask : np.ndarray or None, optional
            A binary mask array of shape (batch_size, n_tokens, n_tokens)
            for object- or background-centric strategies. Values must be
            in [0, 255]. Default is None (ignored for 'random' mode).

        Returns
        -------
        torch.Tensor
            Tensor containing only the kept tokens (with prefix tokens
            re-attached if num_prefix_tokens > 0).
        tuple (torch.Tensor, torch.Tensor), optional
            If `return_indices` is True, also returns the keep_indices
            tensor of shape (batch_size, num_kept_tokens).
        """
        #if not self.training or self.prob == 0.:
        if self.prob == 0.: # We wish to be able to do patch dropout during inference
            if self.return_indices:
                return x, None
            return x

        if self.num_prefix_tokens:
            prefix_tokens, x = x[:, :self.num_prefix_tokens], x[:, self.num_prefix_tokens:]
        else:
            prefix_tokens = None

        B = x.shape[0]
        L = x.shape[1]
        num_keep = max(1, int(L * (1. - self.prob)))
        if self.dropout_strategy == 'random':
            keep_indices = torch.argsort(torch.randn(B, L, device=x.device), dim=-1)[:, :num_keep]
        elif self.dropout_strategy == 'object-centric':
            keep_indices = self.get_object_indices(x, mask, B, num_keep)
        elif self.dropout_strategy == 'background-centric':
            keep_indices = self.get_background_indices(x, mask, B, num_keep)
        elif self.dropout_strategy == 'grid':
            # TODO: Implement grid dropout strategy
            keep_indices = self.get_grid_indices(x, mask, B, num_keep)
        else:
            raise ValueError(f"Unknown dropout strategy: {self.dropout_strategy}")
        if self.ordered:
            # NOTE does not need to maintain patch order in typical transformer use,
            # but possibly useful for debug / visualization
            keep_indices = keep_indices.sort(dim=-1)[0]
        x = x.gather(1, keep_indices.unsqueeze(-1).expand((-1, -1) + x.shape[2:]))

        if prefix_tokens is not None:
            x = torch.cat((prefix_tokens, x), dim=1)

        if self.return_indices:
            return x, keep_indices
        return x