import torch
import logging

from typing import Union, Tuple
from tqdm import tqdm
import numpy as np
import warnings

from . import GRAPPAReconSpec
from .utils import extract_sampled_regions, get_indices_from_mask, pad_back_to_size

logger = logger = logging.getLogger(__name__)


def apply_grappa_kernel(
    sig,
    grappa_recon_spec: GRAPPAReconSpec,
    *,
    batch_size: int = 1,          # NOTE: this is the "y-chunk batch" from your original code
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
        # extract_sampled_regions is assumed batch-ready (B,C,ky,kz,kx)
        sig, start_loc, end_loc = extract_sampled_regions(sig, acs_only=False)

        # Measure shift using batch 0 (assumes same sampling geometry across batch)
        sig_sampled_pat = sig[0].abs().sum(0).sum(-1) != 0  # (ky,kz)
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
        # Use batch 0 to estimate shift (assumes same sampling across batch)
        # original: shift_y, shift_z = abs(sig).sum(0).sum(-1).nonzero()[0]
        shift_y, shift_z = sig[0].abs().sum(0).sum(-1).nonzero()[0].tolist()

    rec = torch.zeros_like(sig)

    # NOTE: batch_size here is "y chunk batch" from original code; avoid confusion with B
    y_chunk_batch = batch_size

    size_chunk_y = sbly + tbly * (y_chunk_batch - 1)
    y_ival = range(shift_y, max(rec.shape[2] - sbly, 1), tbly * y_chunk_batch)  # ky dimension is 2 now
    z_ival = np.arange(0, max(rec.shape[3] - sblz, 1), tblz)                    # kz dimension is 3 now

    if not quiet:
        logger.info("GRAPPA Reconstruction...")

    cnt = shift_z
    idxs_src = idxs_src.flatten()

    for y in tqdm(y_ival, disable=quiet):
        # ---------------------------------------------------------
        # (3) y-slicing now on ky dimension (dim=2): sig[:,:, y: ...]
        # ---------------------------------------------------------
        sig_y = sig[:, :, y : y + size_chunk_y, :, :]
        sig_y = sig_y.cuda() if cuda and cuda_mode in ["all", "application"] else sig_y

        zival = cnt + z_ival
        for z in zival:
            # ---------------------------------------------------------
            # (4) Build blocks with correct dims for batched sig
            # sig_y: (B,C,ky_chunk,kz,kx)
            # slice kz: z:z+sblz -> (B,C,ky_chunk,sblz,kx)
            # unfold ky dim=2 and kx dim=4
            # ---------------------------------------------------------
            blocks = sig_y[:, :, :, z : z + sblz, :].unfold(dimension=2, size=sbly, step=tbly) \
                                                   .unfold(dimension=4, size=sblx, step=tblx)
            # blocks shape: (B, C, nY, sblz, nX, sbly, sblx) or similar depending on unfold ordering
            # Let's permute to: (B, nY, nX, C, sbly, sblz, sblx)
            blocks = blocks.permute(0, 2, 4, 1, 5, 3, 6)

            cur_batch_sz_y = blocks.shape[1]   # nY
            cur_batch_sz_x = blocks.shape[2]   # nX

            blocks = blocks.reshape(B, cur_batch_sz_y, cur_batch_sz_x, nc, -1)[..., idxs_src]

            # ---------------------------------------------------------
            # (5) Fully-sampled test now sums over coil dimension at dim=3
            # blocks: (B, nY, nX, C, nSrc)
            # ---------------------------------------------------------
            are_targets_fully_sampled = (blocks.abs().sum(3) != 0).sum(-1) == idxs_src.sum()  # (B,nY,nX)

            if isGolfSparks:
                if not torch.any(are_targets_fully_sampled):
                    continue

                locs = torch.nonzero(are_targets_fully_sampled, as_tuple=True)  # (b_idx, y_idx, x_idx)

                res = torch.zeros(
                    (B, cur_batch_sz_y, cur_batch_sz_x, nc, tbly, tblz, tblx),
                    dtype=blocks.dtype,
                    device=blocks.device,
                )

                sel = blocks[locs[0], locs[1], locs[2]]  # (N, C, nSrc)
                sel = sel.reshape(sel.shape[0], -1)
                res_vals = (sel @ grappa_kernel).reshape(len(locs[0]), nc, tbly, tblz, tblx)
                res[locs[0], locs[1], locs[2]] = res_vals

            else:
                # Flatten (B*nY*nX, features)
                blocks2 = blocks.reshape(B * cur_batch_sz_y * cur_batch_sz_x, -1)
                res = (blocks2 @ grappa_kernel).reshape(B, cur_batch_sz_y, cur_batch_sz_x, nc, tbly, tblz, tblx)

            # ---------------------------------------------------------
            # (6) Write back into rec with batch dim preserved
            # res: (B,nY,nX,C,tbly,tblz,tblx)
            # want: (B,C,nY*tbly,tblz,nX*tblx)
            # ---------------------------------------------------------
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

    # Keep acquired samples
    rec[sig.abs() > 0] = sig[sig.abs() > 0]

    if isGolfSparks:
        rec = pad_back_to_size(rec, vol_shape, start_loc, end_loc)
    else:
        # Remove padding (ky dim is 2, kz dim is 3, kx dim is 4 now)
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

    # ---------------------------------------------------------
    # (7) Preserve old API: if input was unbatched, squeeze batch dim back out
    # ---------------------------------------------------------
    if squeeze_batch:
        rec = rec.squeeze(0)

    return rec, grappa_recon_spec
