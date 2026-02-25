import torch
import logging
from typing import Union

from . import GRAPPAReconSpec
from .estimation import estimate_grappa_kernel
from .application import apply_grappa_kernel
from .utils import extract_sampled_regions

logger = logging.getLogger(__name__)


def GRAPPA_Recon(
    sig: torch.Tensor,
    acs: torch.Tensor,
    af: Union[list[int], tuple[int, ...]] = None,
    delta: int = 0,
    kernel_size: Union[list[int], tuple[int, ...]] = (4, 4, 5),
    lambda_: float = 1e-4,
    batch_size: int = 1,
    grappa_recon_spec: GRAPPAReconSpec = None,
    mask: torch.Tensor = None,
    isGolfSparks: bool = False,
    cuda: bool = True,
    cuda_mode: str = "all",
    return_kernel: bool = False,
    quiet: bool = False,
):
    """
    Supports:
      sig: (C,ky,kz,kx) or (B,C,ky,kz,kx)
      acs: (C,acsky,acskz,acskx) or (B,C,acsky,acskz,acskx) or None

    If you want one kernel per patch, you must pass batched ACS (or let it be extracted from batched sig).
    """

    # Determine batch size of sig (patch batch)
    sig_B = sig.shape[0] if sig.ndim == 5 else 1

    if grappa_recon_spec is None:
        if acs is None:
            # Must be the batch-aware extract_sampled_regions
            acs = extract_sampled_regions(sig, acs_only=True)

        # Consistency check if ACS is batched
        acs_B = acs.shape[0] if acs.ndim == 5 else 1
        if sig_B != acs_B and acs.ndim == 5:
            raise ValueError(f"Batch mismatch: sig has B={sig_B}, acs has B={acs_B}. One kernel per patch requires matching batches.")

        grappa_recon_spec = estimate_grappa_kernel(
            acs,
            af=af,
            kernel_size=kernel_size,
            delta=delta,
            lambda_=lambda_,
            cuda=cuda,
            cuda_mode=cuda_mode,
            isGolfSparks=isGolfSparks,
            quiet=quiet,
        )

    # If you provide a spec, enforce its batch matches sig when spec is batched
    if hasattr(grappa_recon_spec, "weights"):
        W = grappa_recon_spec.weights
        if sig.ndim == 5 and W.ndim == 3 and W.shape[0] != sig.shape[0]:
            raise ValueError(f"Kernel batch mismatch: sig has B={sig.shape[0]} but spec.weights has B={W.shape[0]}.")

    return apply_grappa_kernel(
        sig,
        grappa_recon_spec,
        batch_size=batch_size,
        cuda=cuda,
        cuda_mode=cuda_mode,
        mask=mask,
        isGolfSparks=isGolfSparks,
        return_kernel=return_kernel,
        quiet=quiet,
    )
