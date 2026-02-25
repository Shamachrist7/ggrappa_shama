import torch
import logging

from . import GRAPPAReconSpec
from .utils import get_src_tgs_blocks

logger = logging.getLogger(__name__)


def estimate_grappa_kernel(
    acs,
    af,
    kernel_size=[4, 4, 5],
    delta=0,
    lambda_=1e-4,
    cuda=False,
    cuda_mode="estimation",
    isGolfSparks=False,
    quiet=False,
    dtype=torch.complex64,
) -> GRAPPAReconSpec:

    if len(af) == 1:
        af = [af[0], 1]

    # Normalize ACS -> (B, C, ky, kz, kx)
    if acs.ndim == 3:
        acs = acs[:, :, None, :]   # (C,ky,kx)->(C,ky,1,kx)
    squeeze_batch = False
    if acs.ndim == 4:
        acs = acs.unsqueeze(0)
        squeeze_batch = True
    elif acs.ndim != 5:
        raise ValueError(f"acs must be (C,ky,kz,kx) or (B,C,ky,kz,kx) (or legacy (C,ky,kx)). Got {acs.shape}")

    acs = acs.to(dtype)
    B, nc, acsny, acsnz, acsnx = acs.shape

    # Pattern
    pat = torch.zeros([((k - 1) * af[i] * [1, 1][i == 0] + 1) for i, k in enumerate(kernel_size[:2])])

    cnt = 0
    for y in range(0, pat.shape[0], af[0]):
        pat[y, cnt::af[1]] = 1
        cnt = (cnt + delta) % af[1]

    tbly, tblz, tblx = af[0], af[1], 1

    sbly = min(pat.shape[0], acsny)
    sblz = min(pat.shape[1], acsnz)
    sblx = min(kernel_size[-1], acsnx)

    xpos = (sblx - 1) // 2
    ypos = (sbly - tbly) // 2
    zpos = (sblz - tblz) // 2

    idxs_src = (pat[:sbly, :sblz] == 1)
    idxs_src = idxs_src.unsqueeze(-1).expand(*idxs_src.size(), sblx)

    idxs_tgs = torch.zeros(sbly, sblz, sblx, dtype=torch.bool, device=acs.device)
    idxs_tgs[ypos:ypos + tbly, zpos:zpos + tblz, xpos:xpos + 1] = True

    nsp = int(idxs_src.sum().item())
    ntg = int(idxs_tgs.sum().item())

    if not quiet:
        logger.info("GRAPPA Kernel estimation (per patch)...")

    # Unfold: ky dim=2, kz dim=3, kx dim=4
    blocks = acs.unfold(dimension=2, size=sbly, step=1) \
                .unfold(dimension=3, size=sblz, step=1) \
                .unfold(dimension=4, size=sblx, step=1)
    # blocks: (B, C, nY, nZ, nX, sbly, sblz, sblx)

    if cuda and cuda_mode in ["all", "estimation"]:
        blocks = blocks.cuda()
        idxs_src = idxs_src.cuda()
        idxs_tgs = idxs_tgs.cuda()

    # Per-patch kernels
    if isGolfSparks:
        # Now batched helper: returns lists
        src_list, tgs_list = get_src_tgs_blocks(blocks, idxs_src, idxs_tgs, check_type='acs')

        kernels = []
        for b in range(B):
            src_b, tgs_b = src_list[b], tgs_list[b]  # (C,N,nsp), (C,N,ntg)
            if src_b.shape[1] == 0:
                # No valid blocks: return zeros kernel (or you can raise)
                F = nc * nsp
                G = nc * ntg
                kernels.append(torch.zeros((F, G), dtype=acs.dtype, device=blocks.device))
                continue

            X = src_b.permute(1, 0, 2).reshape(-1, nc * nsp)  # (N, F)
            Y = tgs_b.permute(1, 0, 2).reshape(-1, nc * ntg)  # (N, G)

            XtX = X.conj().T @ X
            XtY = X.conj().T @ Y
            Fdim = XtX.shape[0]
            XtX = XtX + lambda_ * torch.eye(Fdim, dtype=XtX.dtype, device=XtX.device)
            W = torch.linalg.solve(XtX, XtY)  # (F, G)
            kernels.append(W)

        grappa_kernel = torch.stack(kernels, dim=0)  # (B, F, G)

    else:
        # Vectorized non-golf path
        src = blocks[..., idxs_src].reshape(B, nc, -1, nsp)
        tgs = blocks[..., idxs_tgs].reshape(B, nc, -1, ntg)

        X = src.permute(0, 2, 1, 3).reshape(B, -1, nc * nsp)  # (B,N,F)
        Y = tgs.permute(0, 2, 1, 3).reshape(B, -1, nc * ntg)  # (B,N,G)

        XtX = X.conj().transpose(1, 2) @ X                   # (B,F,F)
        XtY = X.conj().transpose(1, 2) @ Y                   # (B,F,G)

        Fdim = XtX.shape[-1]
        I = torch.eye(Fdim, dtype=XtX.dtype, device=XtX.device).unsqueeze(0)
        XtX = XtX + lambda_ * I

        grappa_kernel = torch.linalg.solve(XtX, XtY)          # (B,F,G)

    return GRAPPAReconSpec(
        weights=grappa_kernel,   # NOTE: now (B, F, G)
        af=af,
        delta=delta,
        pos=[ypos, zpos, xpos],
        sbl=[sbly, sblz, sblx],
        tbl=[tbly, tblz, tblx],
        idxs_src=idxs_src,
    )
