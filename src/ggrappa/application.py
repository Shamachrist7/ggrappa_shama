import torch
import logging

from typing import Union, Tuple
from tqdm import tqdm
import numpy as np
import warnings

from . import GRAPPAReconSpec
from .utils import extract_sampled_regions, get_indices_from_mask, pad_back_to_size

logger = logging.getLogger(__name__)


def apply_grappa_kernel(
    sig,
    grappa_recon_spec: GRAPPAReconSpec,
    *,
    batch_size: int = 1,          # this is your y-chunk batching, NOT the patch batch dim
    isGolfSparks: bool = False,
    mask=None,
    cuda: bool = False,
    cuda_mode: str = "all",
    return_kernel: bool = False,
    quiet: bool = False,
    dtype=torch.complex64,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    grappa_kernel = (
        grappa_recon_spec.weights.cuda()
        if cuda and cuda_mode in ["all", "application"]
        else grappa_recon_spec.weights
    )

    ypos, zpos, xpos = grappa_recon_spec.pos
    tbly, tblz, tblx = grappa_recon_spec.tbl
    sbly, sblz, sblx = grappa_recon_spec.sbl
    af = grappa_recon_spec.af
    delta = grappa_recon_spec.delta
    idxs_src = grappa_recon_spec.idxs_src

    sig = sig.to(dtype)

    # ---------------------------------------------------------
    # (1) Support both unbatched (C,ky,kz,kx) and batched (B,C,ky,kz,kx)
    # ---------------------------------------------------------
    squeeze_batch = False
    if sig.ndim == 4:
        sig = sig.unsqueeze(0)   # (1,C,ky,kz,kx)
        squeeze_batch = True
    elif sig.ndim != 5:
        raise ValueError(f"sig must have shape (C,ky,kz,kx) or (B,C,ky,kz,kx). Got {sig.shape}")

    B, nc, ky, kz, kx = sig.shape
    vol_shape = (ky, kz, kx)

    # ---------------------------------------------------------
    # (1b) Normalize kernel shape:
    #   - allow (F,G)  -> expand to (B,F,G)
    #   - allow (B,F,G) -> use as-is
    # ---------------------------------------------------------
    if grappa_kernel.ndim == 2:
        grappa_kernel = grappa_kernel.unsqueeze(0).expand(B, -1, -1).contiguous()
    elif grappa_kernel.ndim == 3:
        if grappa_kernel.shape[0] != B:
            raise ValueError(f"Kernel batch {grappa_kernel.shape[0]} must match sig batch {B}.")
    else:
        raise ValueError(f"grappa_kernel must have shape (F,G) or (B,F,G). Got {grappa_kernel.shape}")

    # ---------------------------------------------------------
    # (2) Mask handling (broadcast mask to (B,C,ky,kz,kx) if needed)
    # ---------------------------------------------------------
    if mask is not None:
        sig_ = sig
        left, size = get_indices_from_mask(mask)

        if mask.ndim == 3:
            mask_bc = mask[None, None, ...]
        elif mask.ndim == 5:
            mask_bc = mask
        else:
            raise ValueError(f"mask must have shape (ky,kz,kx) or (B,C,ky,kz,kx). Got {mask.shape}")

        sig = (sig * mask_bc)[
            ...,
            left[0] : left[0] + size[0],
            left[1] : left[1] + size[1],
            left[2] : left[2] + size[2],
        ]

    shift_y, shift_z = 0, 0

    if isGolfSparks:
        sig, start_loc, end_loc = extract_sampled_regions(sig, acs_only=False)

        # shift estimation uses batch 0 (assumes same sampling geometry across batch)
        sig_sampled_pat = sig[0].abs().sum(0).sum(-1) != 0
        kernel_pat = idxs_src[..., 0]
        y_multi = sig_sampled_pat.shape[0] // tbly // 2
        z_multi = sig_sampled_pat.shape[1] // tblz // 2

        cnt = 0
        found = False
        for shift_y in range(af[0]):
            for shift_z in range(af[1]):
                if torch.all(
                    sig_sampled_pat[
                        y_multi * tbly + shift_y : y_multi * tbly + shift_y + sbly,
                        z_multi * tblz + shift_z + cnt : z_multi * tblz + shift_z + sblz + cnt,
                    ]
                    == kernel_pat
                ):
                    found = True
                    break
            if found:
                break
            cnt = (cnt + delta) % af[1]

        if not found:
            warnings.warn(
                "Could not find the kernel pattern in the sampled region. Using the center of the sampled region as the kernel pattern."
            )
    else:
        shift_y, shift_z = sig[0].abs().sum(0).sum(-1).nonzero()[0].tolist()

    rec = torch.zeros_like(sig)

    y_chunk_batch = batch_size
    size_chunk_y = sbly + tbly * (y_chunk_batch - 1)
    y_ival = range(shift_y, max(rec.shape[2] - sbly, 1), tbly * y_chunk_batch)
    z_ival = np.arange(0, max(rec.shape[3] - sblz, 1), tblz)

    if not quiet:
        logger.info("GRAPPA Reconstruction...")

    cnt = shift_z
    idxs_src_flat = idxs_src.flatten()
    nsrc = int(idxs_src_flat.sum().item())
    Fdim = nc * nsrc  # feature dimension expected by kernel

    for y in tqdm(y_ival, disable=quiet):
        sig_y = sig[:, :, y : y + size_chunk_y, :, :]
        sig_y = sig_y.cuda() if cuda and cuda_mode in ["all", "application"] else sig_y

        zival = cnt + z_ival
        for z in zival:
            blocks = sig_y[:, :, :, z : z + sblz, :].unfold(dimension=2, size=sbly, step=tbly) \
                                                   .unfold(dimension=4, size=sblx, step=tblx)
            blocks = blocks.permute(0, 2, 4, 1, 5, 3, 6)  # (B,nY,nX,C,sbly,sblz,sblx)

            cur_batch_sz_y = blocks.shape[1]
            cur_batch_sz_x = blocks.shape[2]

            blocks = blocks.reshape(B, cur_batch_sz_y, cur_batch_sz_x, nc, -1)[..., idxs_src_flat]
            # blocks: (B,nY,nX,C,nsrc)

            are_targets_fully_sampled = (blocks.abs().sum(3) != 0).sum(-1) == idxs_src_flat.sum()  # (B,nY,nX)

            if isGolfSparks:
                if not torch.any(are_targets_fully_sampled):
                    continue

                res = torch.zeros(
                    (B, cur_batch_sz_y, cur_batch_sz_x, nc, tbly, tblz, tblx),
                    dtype=blocks.dtype,
                    device=blocks.device,
                )

                # Apply correct kernel per batch (ragged selection), loop over batch only
                for b in range(B):
                    mask_b = are_targets_fully_sampled[b]  # (nY,nX)
                    if not torch.any(mask_b):
                        continue
                    loc_yx = torch.nonzero(mask_b, as_tuple=False)  # (Nb,2)
                    sel = blocks[b, loc_yx[:, 0], loc_yx[:, 1]]      # (Nb,C,nsrc)
                    sel = sel.reshape(sel.shape[0], Fdim)            # (Nb,F)
                    out = (sel @ grappa_kernel[b]).reshape(sel.shape[0], nc, tbly, tblz, tblx)
                    res[b, loc_yx[:, 0], loc_yx[:, 1]] = out

            else:
                # Vectorized per-patch application with batched matmul
                # X: (B, N, F), W: (B, F, G) -> Y: (B, N, G)
                X = blocks.reshape(B, cur_batch_sz_y * cur_batch_sz_x, Fdim)
                Y = torch.bmm(X, grappa_kernel)  # (B, N, G)

                res = Y.reshape(B, cur_batch_sz_y, cur_batch_sz_x, nc, tbly, tblz, tblx)

            res_grid = res.permute(0, 3, 1, 4, 5, 2, 6).reshape(
                B, nc, cur_batch_sz_y * tbly, tblz, cur_batch_sz_x * tblx
            )

            rec[
                :, :,
                y + ypos : y + ypos + tbly * cur_batch_sz_y,
                z + zpos : z + zpos + tblz,
                xpos : xpos + tblx * cur_batch_sz_x,
            ] = res_grid

        cnt = (cnt + delta) % af[1]
        del sig_y
        if cuda:
            torch.cuda.empty_cache()

    rec[sig.abs() > 0] = sig[sig.abs() > 0]

    if isGolfSparks:
        rec = pad_back_to_size(rec, vol_shape, start_loc, end_loc)
    else:
        if sbly > 1:
            rec = rec[:, :, (af[0] - ypos) % tbly + ypos : -(sbly - ypos - tbly), :, :]
        if sblz > 1:
            rec = rec[:, :, :, (af[1] - zpos) % tblz + zpos : -(sblz - zpos - tblz), :]
        if sblx > 1:
            rec = rec[:, :, :, :, xpos : -(sblx - xpos - tblx)]

    if mask is not None:
        if mask.ndim == 3:
            mask_crop = mask[left[0] : left[0] + size[0],
                             left[1] : left[1] + size[1],
                             left[2] : left[2] + size[2]]
            mask_crop_bc = mask_crop[None, None, ...]
        else:
            mask_crop_bc = mask[
                ...,
                left[0] : left[0] + size[0],
                left[1] : left[1] + size[1],
                left[2] : left[2] + size[2],
            ]

        rec *= mask_crop_bc
        not_mask = (~mask_crop_bc) if mask_crop_bc.dtype == torch.bool else (1 - mask_crop_bc)

        sig_[..., left[0] : left[0] + size[0],
                 left[1] : left[1] + size[1],
                 left[2] : left[2] + size[2]] = (
            (sig_ * not_mask)[..., left[0] : left[0] + size[0],
                                   left[1] : left[1] + size[1],
                                   left[2] : left[2] + size[2]] + rec
        )
        rec = sig_

    if not quiet:
        logger.info("GRAPPA Reconstruction done.")

    if squeeze_batch:
        rec = rec.squeeze(0)

    return rec, grappa_recon_spec
