import torch
import numpy as np
import scipy as sp


def rss(data, axis=0):
    return np.sqrt(np.sum(np.abs(data)**2, axis=axis))


def get_indices_from_mask(mask):
    if not isinstance(mask, np.ndarray):
        mask = mask.numpy()
    nonzero_indices = np.nonzero(mask)

    min_indices = np.min(nonzero_indices, axis=1)
    max_indices = np.max(nonzero_indices, axis=1)

    cube_dimensions = max_indices - min_indices + 1

    return min_indices, cube_dimensions


def get_src_tgs_blocks(blocks, idxs_src, idxs_tgs, check_type='acs'):
    """
    Supports:
      blocks shape:
        - (C, nY, nZ, nX, sbly, sblz, sblx)
        - (B, C, nY, nZ, nX, sbly, sblz, sblx)

    Returns:
      If unbatched: (src, tgs) with shapes (C, N, nsp) and (C, N, ntg)
      If batched: (src_list, tgs_list), lists of length B where each element is
                  (C, N_b, nsp) and (C, N_b, ntg)
    """
    if blocks.ndim == 7:
        # Unbatched: (C, nY, nZ, nX, sbly, sblz, sblx)
        C = blocks.shape[0]
        blocks_u = blocks
        batched = False
    elif blocks.ndim == 8:
        # Batched: (B, C, nY, nZ, nX, sbly, sblz, sblx)
        batched = True
    else:
        raise ValueError(f"blocks must be 7D or 8D. Got shape {blocks.shape}")

    nsp = int(idxs_src.sum().item())
    ntg = int(idxs_tgs.sum().item())

    def _select_one(blocks_one):
        # blocks_one: (C, nY, nZ, nX, sbly, sblz, sblx)
        if check_type == 'acs':
            # Count nonzero entries per block (across coils), require full sampling in the whole block
            # blocks_one.sum(dim=0): (nY,nZ,nX,sbly,sblz,sblx)
            samples_per_block = (blocks_one.sum(dim=0) != 0).sum(dim=(-3, -2, -1))  # (nY,nZ,nX)
            locy, locz, locx = torch.nonzero(samples_per_block == idxs_src.numel(), as_tuple=True)

        elif check_type == 'all_sampled_srcs':
            # Require: all src locations present AND at least one target missing (common “need to reconstruct” criterion)
            srcs_per_block = blocks_one[..., idxs_src].abs().sum(dim=-1)  # (C,nY,nZ,nX,nsp)->sum over nsp => (C,nY,nZ,nX)
            tgs_per_block  = blocks_one[..., idxs_tgs].abs().sum(dim=-1)  # (C,nY,nZ,nX,ntg)->sum over ntg => (C,nY,nZ,nX)

            # Aggregate across coils to decide presence
            src_ok = (srcs_per_block.sum(dim=0) != 0)  # (nY,nZ,nX)
            tgs_ok = (tgs_per_block.sum(dim=0) != 0)   # (nY,nZ,nX)

            # all sampled srcs AND (not fully sampled targets)
            cond = src_ok & (~tgs_ok)
            locy, locz, locx = torch.nonzero(cond, as_tuple=True)

        else:
            raise ValueError(f"Unknown check_type: {check_type}")

        if locy.numel() == 0:
            # Return empty tensors with correct ranks
            src_empty = blocks_one.new_empty((C, 0, nsp))
            tgs_empty = blocks_one.new_empty((C, 0, ntg))
            return src_empty, tgs_empty

        select_blocks = blocks_one[:, locy, locz, locx]  # (C, N, sbly, sblz, sblx)
        return select_blocks[..., idxs_src], select_blocks[..., idxs_tgs]  # (C,N,nsp), (C,N,ntg)

    if not batched:
        return _select_one(blocks_u)

    # Batched: return ragged lists
    B = blocks.shape[0]
    src_list, tgs_list = [], []
    for b in range(B):
        src_b, tgs_b = _select_one(blocks[b])
        src_list.append(src_b)
        tgs_list.append(tgs_b)
    return src_list, tgs_list

def get_grappa_filled_data_and_loc(sig, rec, params):
    #rec[:, np.abs(sig).sum(axis=0)!=0] = 0
    sampled_mask = np.abs(rec).sum(axis=0) != 0
    extra_data = rec[:, sampled_mask]
    rec_loc = np.nonzero(sampled_mask)
    rec_loc = np.asarray(rec_loc).T
    extra_loc = rec_loc / params['img_size'] - 0.5
    return extra_loc, extra_data
     

def get_cart_portion_sparkling(kspace_shots, traj_params, kspace_data, calc_osf_buffer=10):
    """
    Same behavior as before for kspace_data shaped [C, ...],
    but now also supports batched kspace_data shaped [B, C, ...].

    Returns
    -------
    gridded_data : np.ndarray
        [C, *img_size] if input was [C, ...]
        [B, C, *img_size] if input was [B, C, ...]
    new_kspace_data : np.ndarray
        [C, ...] stacked across rows (legacy behavior) or [B, C, ...] if batched
    new_kspace_loc : np.ndarray
        Same as before (trajectory locations), independent of batch
    """

    # ----------------------------
    # (1) Normalize input shape: force batch dim
    # ----------------------------
    batched = (kspace_data.ndim >= 2)  # always true, but we check whether it already has batch
    if kspace_data.ndim >= 3:
        # Assume [B, C, ...]
        B, C = kspace_data.shape[0], kspace_data.shape[1]
        kspace_data_bc = kspace_data
        squeeze_batch = False
    else:
        # Assume [C, ...] -> make it [1, C, ...]
        C = kspace_data.shape[0]
        B = 1
        kspace_data_bc = kspace_data[None, ...]
        squeeze_batch = True

    # ----------------------------
    # Original logic (trajectory-only parts unchanged)
    # ----------------------------
    grads = np.diff(kspace_shots, axis=1)

    # ----------------------------
    # (2) Reshape k-space data with batch: [B, C, rows, cols]
    #     Original: [C, rows, cols]
    # ----------------------------
    re_kspace_data = kspace_data_bc.reshape(B, C, *kspace_shots.shape[:2])

    mask = grads[..., 1] == 0
    pad_mask = np.pad(mask, ((0, 0), (1, 1)), constant_values=False)
    mask = np.diff(pad_mask * 1)
    starts = np.argwhere(mask == 1)
    ends = np.argwhere(mask == -1)

    osf = 1 / np.mean(
        np.diff(
            kspace_shots[
                kspace_shots.shape[0] // 2,
                kspace_shots.shape[1] // 2 - calc_osf_buffer : kspace_shots.shape[1] // 2 + calc_osf_buffer,
                0,
            ]
        )
        * traj_params["img_size"][0]
    )

    max_length = np.zeros(grads.shape[0]) + osf  # ensure >= 1 point after resampling
    locs = np.ones((grads.shape[0], 2)) * -1
    sampled_loc = [[],] * grads.shape[0]
    cart_loc = [[],] * grads.shape[0]

    new_kspace_data = []
    new_kspace_loc = []

    # ----------------------------
    # (3) gridded_data now includes batch dim: [B, C, *img_size]
    #     Original: [C, *img_size]
    # ----------------------------
    gridded_data = np.zeros((B, C, *traj_params["img_size"]), dtype=np.complex64)

    for start, end in zip(starts, ends):
        row, start_col = start
        _, end_col = end
        length = end_col - start_col
        if length > max_length[row]:
            max_length[row] = length
            locs[row, 0] = start_col
            locs[row, 1] = end_col
            cart_loc[row] = np.copy(kspace_shots[row, start_col:end_col])
            sampled_loc[row] = [start_col, end_col]

    for row, (locs_row, s_loc) in enumerate(zip(cart_loc, sampled_loc)):
        if (
            not len(locs_row)
            or int(
                (s_loc[1] - s_loc[0])
                * np.diff(kspace_shots[row, s_loc[0] : s_loc[0] + 2, 0])
                * traj_params["img_size"][0]
            )
            == 0
        ):
            # ----------------------------
            # (4) Batched append: re_kspace_data[:, :, row] instead of re_kspace_data[:, row]
            # ----------------------------
            new_kspace_data.append(re_kspace_data[:, :, row])  # [B, C, cols]
            new_kspace_loc.append(kspace_shots[row, ...])
            continue

        n_resamp = int(
            (s_loc[1] - s_loc[0])
            * np.diff(kspace_shots[row, s_loc[0] : s_loc[0] + 2, 0])
            * traj_params["img_size"][0]
        )

        # ----------------------------
        # (5) Resample along last axis, preserving [B, C, ...]
        #     Original was [C, ...]
        # ----------------------------
        data = sp.signal.resample(
            re_kspace_data[:, :, row, s_loc[0] : s_loc[1]],  # [B, C, seglen]
            n_resamp,
            axis=-1,
        )  # -> [B, C, n_resamp]

        new_kspace_data.append(
            np.concatenate(
                [re_kspace_data[:, :, row, : s_loc[0]], re_kspace_data[:, :, row, s_loc[1] :]],
                axis=-1,
            )
        )  # [B, C, cols - seglen]

        new_kspace_loc.append(
            np.concatenate([kspace_shots[row, : s_loc[0]], kspace_shots[row, s_loc[1] :]], axis=0)
        )

        # grid placement (same as before)
        locs_grid = locs_row + 0.5
        locs_grid *= np.asarray(traj_params["img_size"]).T
        rounded_locs = locs_grid.round(0).astype("int")

        r0 = rounded_locs[0][0]
        r1 = r0 + data.shape[-1]
        c1 = rounded_locs[0][1]
        c2 = rounded_locs[0][2]

        # ----------------------------
        # (6) Assign batched data: gridded_data[:, :, ...] = data
        #     Original: gridded_data[:, ...] = data
        # ----------------------------
        gridded_data[:, :, r0:r1, c1, c2] = data

    # ----------------------------
    # (7) Stack new_kspace_data into array
    #     We have a list of [B, C, L_row] with varying L_row, so legacy code used hstack.
    #     We'll keep the same behavior: concatenate over last axis.
    # ----------------------------
    new_kspace_data_arr = np.concatenate(new_kspace_data, axis=-1) if len(new_kspace_data) else np.empty((B, C, 0))
    new_kspace_loc_arr = np.concatenate(new_kspace_loc, axis=0) if len(new_kspace_loc) else np.empty((0,) + kspace_shots.shape[1:])

    # ----------------------------
    # (8) If input was unbatched, squeeze batch dim to preserve old API
    # ----------------------------
    if squeeze_batch:
        gridded_data = gridded_data[0]
        new_kspace_data_arr = new_kspace_data_arr[0]

    return gridded_data, new_kspace_data_arr, new_kspace_loc_arr
    
    
def pad_back_to_size(sig: torch.Tensor, vol_shape, start_loc, end_loc):
    """
    Pads sig back to vol_shape.

    Supports:
      - sig shape (C, ky, kz, kx)  -> returns (C, ky, kz, kx)
      - sig shape (B, C, ky, kz, kx) -> returns (B, C, ky, kz, kx)
    """
    ky, kz, kx = vol_shape
    start_ky = ky // 2
    start_kz = kz // 2
    start_kx = kx // 2

    squeeze_batch = False
    if sig.ndim == 4:
        # (C, ky, kz, kx) -> (1, C, ky, kz, kx)
        sig = sig.unsqueeze(0)
        squeeze_batch = True
    elif sig.ndim != 5:
        raise ValueError(f"sig must be 4D or 5D. Got shape {sig.shape}")

    B, C, _, _, _ = sig.shape

    rec = torch.zeros(
        (B, C, ky, kz, kx),
        dtype=sig.dtype,
        device=sig.device,
    )

    y0 = start_ky - start_loc[0] + 1
    y1 = start_ky + end_loc[0]   - 1
    z0 = start_kz - start_loc[1] + 1
    z1 = start_kz + end_loc[1]   - 1
    x0 = start_kx - start_loc[2] + 1
    x1 = start_kx + end_loc[2]   - 1

    rec[:, :, y0:y1, z0:z1, x0:x1] = sig

    if squeeze_batch:
        rec = rec.squeeze(0)

    return rec


def extract_sampled_regions(sig: torch.Tensor, acs_only: bool = True):
    """Extracts the Auto-Calibration Signal (ACS) region from the input signal.

    Now supports batched input:
        sig shape: (B, C, ky, kz, kx)

    Notes
    -----
    This keeps the original behavior of computing the ACS box using coil 0 and
    the center line. The same crop is then applied to all coils and all batches.
    """
    # sig: (B, C, ky, kz, kx)
    B, C, ky, kz, kx = sig.shape

    start_ky = ky // 2
    start_kz = kz // 2
    start_kx = kx // 2

    torch_fn = torch.min
    if acs_only:
        sig_ = (sig.abs() == 0)
    else:
        sig_ = (sig.abs() != 0)
        torch_fn = torch.max

    # Use batch 0, coil 0 to determine bounds (same idea as original: coil 0).
    # Slices are now indexed as: sig_[b, c, ky, kz, kx]
    left_start_ky = torch_fn(torch.nonzero(sig_[0, 0, start_ky:, start_kz, start_kx], as_tuple=False)).item()
    left_end_ky   = torch_fn(torch.nonzero(sig_[0, 0, :start_ky+1, start_kz, start_kx].flip(0), as_tuple=False)).item()

    left_start_kz = torch_fn(torch.nonzero(sig_[0, 0, start_ky, start_kz:, start_kx], as_tuple=False)).item()
    left_end_kz   = torch_fn(torch.nonzero(sig_[0, 0, start_ky, :start_kz+1, start_kx].flip(0), as_tuple=False)).item()

    left_start_kx = torch_fn(torch.nonzero(sig_[0, 0, start_ky, start_kz, start_kx:], as_tuple=False)).item()
    left_end_kx   = torch_fn(torch.nonzero(sig_[0, 0, start_ky, start_kz, :start_kx+1].flip(0), as_tuple=False)).item()

    # Compute slice bounds exactly as before
    ky0 = start_ky - left_start_ky + 1
    ky1 = start_ky + left_end_ky   - 1
    kz0 = start_kz - left_start_kz + 1
    kz1 = start_kz + left_end_kz   - 1
    kx0 = start_kx - left_start_kx + 1
    kx1 = start_kx + left_end_kx   - 1

    center = sig[:, :, ky0:ky1, kz0:kz1, kx0:kx1]

    if acs_only:
        return center

    start_loc = (left_start_ky, left_start_kz, left_start_kx)
    end_loc   = (left_end_ky,   left_end_kz,   left_end_kx)
    return center, start_loc, end_loc
    

def pinv_batch(M, lambda_=1e-4, cuda=True):
    if cuda: M = M.cuda()
    MM = M.H @ M
    del M
    #torch.cuda.empty_cache()

    # Might also consider Power Iteration methods to speedup the process for large M matrix?
    S = torch.linalg.eigvalsh(MM)[-1].item()

    MM = MM.cpu()
    regularizer = (lambda_**2) * abs(S) * torch.eye(MM.shape[0], device=MM.device)
    del S
    #torch.cuda.empty_cache()
    reg_pinv = torch.linalg.pinv(MM + regularizer)
    del MM
    return reg_pinv

def pinv(M, lambda_=1e-4):
    if M.shape[0] > M.shape[1]:
        MM = M.H @ M
        finalTranspose = False
    else:
        MM = M @ M.H
        finalTranspose = True
    S = torch.linalg.eigvalsh(MM)[-1].item()
    regularizer = (lambda_**2) * abs(S) * torch.eye(MM.shape[0], device=M.device)
    reg_pinv = torch.linalg.pinv(MM + regularizer)
    return reg_pinv.H if finalTranspose else reg_pinv


def pinv_linalg_batch(A, lamdba_=1e-4, cuda=True):
    if cuda: A = A.cuda()
    AA = A.H@A
    del A
    #torch.cuda.empty_cache()
    S = torch.linalg.eigvalsh(AA)[-1].item() # Largest eigenvalue
    lambda_sq = (lamdba_**2) * abs(S)
    del S
    #torch.cuda.empty_cache()
    I = torch.eye(AA.shape[0], dtype=AA.dtype, device=AA.device)
    regularized_matrix = AA + I * lambda_sq
    del I, AA, lambda_sq
    #torch.cuda.empty_cache()
    A = A.cpu()
    regularized_matrix = regularized_matrix.cpu()
    return torch.linalg.solve(regularized_matrix, A.H)


def pinv_linalg(A, lamdba_=1e-4):
    m,n = A.shape
    if n > m:
        AA = A@A.H
    else:
        AA = A.H@A
    S = torch.linalg.eigvalsh(AA)[-1].item()
    lambda_sq = (lamdba_**2) * abs(S)

    I = torch.eye(AA.shape[0], dtype=A.dtype, device=A.device)

    regularized_matrix = AA + I * lambda_sq

    return torch.linalg.solve(regularized_matrix, A.H)
