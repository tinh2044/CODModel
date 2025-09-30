import numpy as np

from scipy.ndimage import convolve
from scipy.ndimage import distance_transform_edt as _bwdist

_EPS = np.spacing(1)


def _to_numpy(pred, gt):
    """
    Convert tensors (B,1,H,W) to numpy arrays (B,H,W) without copying unnecessarily.
    """
    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred.squeeze(1)
    if gt.dim() == 4 and gt.size(1) == 1:
        gt = gt.squeeze(1)
    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    return pred_np, gt_np


def _prepare_data_numpy(pred, gt, normalize=True):
    """
    Prepare data to align with the classic SOD evaluation convention:
    - pred64 in [0,1] and min-max normalized per-sample if non-constant
    - gt: boolean mask via threshold 128 (if input is uint8-like) or 0.5 otherwise
    Accepts (B,H,W) arrays and processes per-sample.
    """
    B = pred.shape[0]
    pred_out = np.empty_like(pred, dtype=np.float64)
    gt_out = np.empty_like(gt, dtype=bool)

    # Heuristics: if pred max > 1.0, assume 0..255 and scale first
    if normalize:
        # normalize per-sample similar to py_sod_metrics.utils.prepare_data
        for b in range(B):
            p = pred[b]
            if p.dtype != np.float32 and p.dtype != np.float64:
                p = p.astype(np.float64)
            # scale to [0,1]
            if p.max() > 1.0:
                p = p / 255.0
            # min-max per image if not constant
            p_max, p_min = p.max(), p.min()
            if p_max != p_min:
                p = (p - p_min) / (p_max - p_min)
            pred_out[b] = p

            g = gt[b]
            # choose threshold rule based on value range
            if g.max() > 1.0:
                gt_out[b] = g > 128
            else:
                gt_out[b] = g > 0.5
    else:
        # validate
        if pred.dtype not in (np.float32, np.float64):
            raise TypeError(
                f"Prediction array must be float32 or float64, got {pred.dtype}"
            )
        if pred.min() < 0 or pred.max() > 1:
            raise ValueError("Prediction values must be in range [0, 1]")
        if gt.dtype == bool:
            gt_out = gt
        else:
            gt_out = gt > 0.5
        pred_out = pred.astype(np.float64, copy=False)

    return pred_out, gt_out


def _adaptive_threshold(x, max_value=1.0):
    return float(min(2.0 * float(x.mean()), max_value))


def mae(pred_np, gt_np):
    """
    Batch MAE for SOD.
    Args:
        pred, gt: tensors of shape (B,1,H,W) or (B,H,W)
    Returns:
        Mean MAE over batch (float)
    """

    maes = np.mean(np.abs(pred_np - gt_np), axis=(1, 2))
    return float(maes.mean())


def f_measure(pred_np, gt_np, beta=0.3):
    """
    F-measure with adaptive version and dynamic curve over thresholds 0..255.
    Returns dict with:
        - 'adp' mean adaptive F-measure over batch
        - 'curve'(256,) mean curve over batch
    """

    B = pred_np.shape[0]
    adaptive_vals = np.zeros((B,), dtype=np.float64)
    curves = np.zeros((B, 256), dtype=np.float64)

    for b in range(B):
        p = pred_np[b]
        g = gt_np[b]

        # Adaptive
        thr = _adaptive_threshold(p, 1.0)
        bin_pred = p >= thr
        inter = (bin_pred & g).sum()
        if inter == 0:
            adaptive_vals[b] = 0.0
        else:
            pre = inter / max(bin_pred.sum(), 1)
            rec = inter / max(g.sum(), 1)
            adaptive_vals[b] = (1 + beta) * pre * rec / (beta * pre + rec)

        # Dynamic 0..255 thresholds (use hist trick)
        p255 = (p * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_hist, _ = np.histogram(p255[g], bins=bins)
        bg_hist, _ = np.histogram(p255[~g], bins=bins)
        fg_w = np.cumsum(np.flip(fg_hist))
        bg_w = np.cumsum(np.flip(bg_hist))
        TPs = fg_w
        Ps = fg_w + bg_w
        Ps[Ps == 0] = 1
        T = max(int(g.sum()), 1)
        precisions = TPs / Ps
        recalls = TPs / T
        num = (1 + beta) * precisions * recalls
        den = np.where(num == 0, 1.0, beta * precisions + recalls)
        curves[b] = num / den

    return {"adp": adaptive_vals.mean(), "curve": curves.mean(axis=0)}


def s_measure(pred_np, gt_np, alpha=0.5):
    """S-measure for SOD (batch-averaged)."""

    def _s_object(x):
        mean = float(x.mean())
        std = float(np.std(x, ddof=1))
        return 2 * mean / (mean * mean + 1 + std + _EPS)

    def _ssim(px, gx):
        h, w = px.shape
        N = h * w
        x = float(px.mean())
        y = float(gx.mean())
        sigma_x = float(((px - x) ** 2).sum() / (N - 1))
        sigma_y = float(((gx - y) ** 2).sum() / (N - 1))
        sigma_xy = float(((px - x) * (gx - y)).sum() / (N - 1))
        a = 4 * x * y * sigma_xy
        b = (x * x + y * y) * (sigma_x + sigma_y)
        if a != 0:
            return a / (b + _EPS)
        return 1.0 if b == 0 else 0.0

    def _region(p, g):
        h, w = g.shape
        area = h * w
        if g.sum() == 0:
            cy, cx = round(h / 2), round(w / 2)
        else:
            cy, cx = np.argwhere(g).mean(axis=0).round()
        cy, cx = int(cy) + 1, int(cx) + 1
        w_lt = cx * cy / area
        w_rt = cy * (w - cx) / area
        w_lb = (h - cy) * cx / area
        w_rb = 1 - w_lt - w_rt - w_lb
        score = 0.0
        score += _ssim(p[0:cy, 0:cx], g[0:cy, 0:cx]) * w_lt
        score += _ssim(p[0:cy, cx:w], g[0:cy, cx:w]) * w_rt
        score += _ssim(p[cy:h, 0:cx], g[cy:h, 0:cx]) * w_lb
        score += _ssim(p[cy:h, cx:w], g[cy:h, cx:w]) * w_rb
        return score

    def _object(p, g):
        gm = g.mean()
        fg_score = _s_object(p[g]) * gm
        bg_score = _s_object((1 - p)[~g]) * (1 - gm)
        return fg_score + bg_score

    vals = []
    for b in range(pred_np.shape[0]):
        p, g = pred_np[b], gt_np[b]
        y = g.mean()
        if y == 0:
            sm = 1 - p.mean()
        elif y == 1:
            sm = p.mean()
        else:
            sm = max(0.0, _object(p, g) * alpha + _region(p, g) * (1 - alpha))
        vals.append(sm)
    return float(np.mean(vals))


def e_measure(pred_np, gt_np):
    """
    E-measure with adaptive version and dynamic curve (0..255).
    Returns dict with 'adp' (float) and 'curve' (np.ndarray(256,)).
    """

    B, H, W = pred_np.shape
    gt_size = H * W
    adaptive_vals = np.zeros((B,), dtype=np.float64)
    curves = np.zeros((B, 256), dtype=np.float64)

    for b in range(B):
        p = pred_np[b]
        g = gt_np[b]
        FG = int(g.sum())

        def _em_for_threshold(bin_pred):
            fg_fg = np.count_nonzero(bin_pred & g)
            fg_bg = np.count_nonzero(bin_pred & (~g))
            pred_fg = fg_fg + fg_bg
            pred_bg = gt_size - pred_fg

            if FG == 0:
                enhanced_sum = pred_bg
            elif FG == gt_size:
                enhanced_sum = pred_fg
            else:
                bg_fg = FG - fg_fg
                bg_bg = pred_bg - bg_fg

                parts = [fg_fg, fg_bg, bg_fg, bg_bg]
                mean_pred = pred_fg / gt_size
                mean_gt = FG / gt_size
                combs = [
                    (1 - mean_pred, 1 - mean_gt),
                    (1 - mean_pred, 0 - mean_gt),
                    (0 - mean_pred, 1 - mean_gt),
                    (0 - mean_pred, 0 - mean_gt),
                ]
                res_parts = []
                for part, (a, b_) in zip(parts, combs):
                    align_v = 2 * (a * b_) / (a * a + b_ * b_ + _EPS)
                    enh_v = (align_v + 1) ** 2 / 4
                    res_parts.append(enh_v * part)
                enhanced_sum = sum(res_parts)
            return enhanced_sum / (gt_size - 1 + _EPS)

        # adaptive
        thr = _adaptive_threshold(p, 1.0)
        adaptive_vals[b] = _em_for_threshold(p >= thr)

        # dynamic curve
        p255 = (p * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        tp_hist, _ = np.histogram(p255[g], bins=bins)
        fp_hist, _ = np.histogram(p255[~g], bins=bins)
        tp_w = np.cumsum(np.flip(tp_hist))
        fp_w = np.cumsum(np.flip(fp_hist))
        pred_fg_w = tp_w + fp_w
        pred_bg_w = gt_size - pred_fg_w

        if FG == 0:
            enhanced_sum = pred_bg_w
        elif FG == gt_size:
            enhanced_sum = pred_fg_w
        else:
            bg_fg_w = FG - tp_w
            bg_bg_w = pred_bg_w - bg_fg_w
            parts_w = [tp_w, fp_w, bg_fg_w, bg_bg_w]
            mean_pred_w = pred_fg_w / gt_size
            mean_gt = FG / gt_size
            combs = [
                (1 - mean_pred_w, 1 - mean_gt),
                (1 - mean_pred_w, 0 - mean_gt),
                (0 - mean_pred_w, 1 - mean_gt),
                (0 - mean_pred_w, 0 - mean_gt),
            ]
            res_parts = np.empty((4, 256), dtype=np.float64)
            for i, (part, (a, b_)) in enumerate(zip(parts_w, combs)):
                align_v = 2 * (a * b_) / (a * a + b_ * b_ + _EPS)
                enh_v = (align_v + 1) ** 2 / 4
                res_parts[i] = enh_v * part
            enhanced_sum = res_parts.sum(axis=0)

        curves[b] = enhanced_sum / (gt_size - 1 + _EPS)

    return {"adp": adaptive_vals.mean(), "curve": curves.mean(axis=0)}


def weighted_f_measure(pred_np, gt_np, beta=1.0):
    """
    Weighted F-measure (Margolin et al. CVPR'14), batch-averaged.
    """
    vals = []
    for b in range(pred_np.shape[0]):
        p = pred_np[b]
        g = gt_np[b]

        # distance transform on background of gt
        Dst, Idx = _bwdist(g == 0, return_indices=True)
        E = np.abs(p - g)
        Et = E.copy()
        # handle edges of foreground region
        Et[g == 0] = Et[Idx[0][g == 0], Idx[1][g == 0]]

        # gaussian smoothing
        def _gauss2d(shape=(7, 7), sigma=5.0):
            m, n = [(ss - 1) / 2 for ss in shape]
            y, x = np.ogrid[-m : m + 1, -n : n + 1]
            h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            h /= h.sum() if h.sum() != 0 else 1
            return h

        K = _gauss2d((7, 7), 5.0)
        EA = convolve(Et, weights=K, mode="constant", cval=0)
        MIN_E_EA = np.where(g & (EA < E), EA, E)

        # pixel importance
        Bmask = np.where(g == 0, 2 - np.exp(np.log(0.5) / 5.0 * Dst), np.ones_like(g))
        Ew = MIN_E_EA * Bmask

        TPw = np.sum(g) - np.sum(Ew[g == 1])
        FPw = np.sum(Ew[g == 0])

        R = 1 - np.mean(Ew[g == 1]) if np.any(g) else 0.0
        P = TPw / (TPw + FPw + _EPS)
        Q = (1 + beta) * R * P / (R + beta * P + _EPS)
        vals.append(Q)
    return float(np.mean(vals))


def calculate_iou(
    pred_mask, gt_mask, threshold= 0.5
):
    """
    Calculate Intersection over Union (IoU) for binary segmentation
    Args:
        pred_mask: (B, 1, H, W) - probabilities [0,1] for positive class
        gt_mask: (B, 1, H, W) - binary mask [0,1]
        threshold: threshold for binary classification
    Returns:
        IoU score
    """
    if pred_mask.dim() == 4:
        pred_mask = pred_mask.squeeze(1)  # (B,1,H,W) -> (B,H,W)
    if gt_mask.dim() == 4:
        gt_mask = gt_mask.squeeze(1)  # (B,1,H,W) -> (B,H,W)

    # pred_mask is already probabilities [0,1], apply threshold
    pred_binary = (pred_mask > threshold).float()
    gt_binary = (gt_mask > threshold).float()

    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum() - intersection

    return (intersection / (union + 1e-8)).item()


def compute_metrics(pred, gt, *, normalize=True):
    """
    Convenience wrapper to compute five SOD metrics on batch tensors.
    Returns a dict with keys: 'mae', 'smeasure', 'fmeasure', 'emeasure', 'weighted_fmeasure'.
    - fmeasure/emeasure have dict values: {'adp', 'curve'(256,)}
    """

    pred_np, gt_np = _to_numpy(pred, gt)
    pred_np, gt_np = _prepare_data_numpy(pred_np, gt_np, normalize)

    em = e_measure(pred_np, gt_np)
    fm = f_measure(pred_np, gt_np)
    return {
        "mae": mae(pred_np, gt_np),
        "sm": s_measure(pred_np, gt_np),
        "wfm": weighted_f_measure(pred_np, gt_np),
        "adpEm": em["adp"],
        "meanEm": em["curve"].mean(),
        "maxEm": em["curve"].max(),
        "adpFm": fm["adp"],
        "meanFm": fm["curve"].mean(),
        "maxFm": fm["curve"].max(),
        "iou": calculate_iou(pred, gt),
    }
