import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-7


def dice_loss(pred_logits, target, eps=1e-6):
    """
    pred_logits: (B,1,H,W) logits
    target: (B,1,H,W) binary {0,1}
    returns dice loss (1 - dice)
    """
    prob = torch.sigmoid(pred_logits)
    num = 2 * (prob * target).sum(dim=(2, 3))
    den = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    loss = 1.0 - (num / den)
    return loss.mean()


bce_loss_fn = nn.BCEWithLogitsLoss()


def bce_dice_loss(pred_logits, target, bce_weight=1.0, dice_weight=1.0):
    """
    combined BCE + Dice
    target: binary (B,1,H,W)
    """
    b = bce_loss_fn(pred_logits, target)
    d = dice_loss(pred_logits, target)
    return bce_weight * b + dice_weight * d


def make_boundary_target(mask, dilation_radius=3):
    """
    mask: (B,1,H,W) binary float tensor {0,1}
    returns boundary GT same size (B,1,H,W) binary float
    morphological gradient = dilate - erode (approx via maxpool/minpool)
    dilation_radius: integer
    """
    assert mask.ndim == 4
    kernel = 2 * dilation_radius + 1
    # max pool approximates dilation, min pool approximate erosion via -max(-x)
    pad = dilation_radius
    # dilated
    dilated = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, kernel_size=kernel, stride=1, padding=pad)
    boundary = (dilated - eroded).clamp(0.0, 1.0)
    return boundary


def sample_positions_from_mask(mask, n_samples, rng=None):
    """
    mask: (H,W) binary torch tensor (0/1)
    returns list of (y,x) coordinates length up to n_samples
    """
    # find positives
    pos = torch.nonzero(mask, as_tuple=False)  # (K,2) with (y,x)
    K = pos.shape[0]
    if K == 0:
        return []
    if K <= n_samples:
        selected = pos
    else:
        # random sample without replacement
        idx = torch.randperm(K)[:n_samples]
        selected = pos[idx]
    # convert to python list tuples
    coords = [(int(p[0].item()), int(p[1].item())) for p in selected]
    return coords


def sample_hard_negatives(pred_prob_map, gt_mask, n_hard):
    """
    pred_prob_map: (H,W) predicted probability (0..1) for foreground, torch tensor
    gt_mask: (H,W) binary (0/1)
    return list of coords of background positions with high pred prob
    """
    # mask background positions
    bg_mask = (gt_mask == 0).float()
    scores = pred_prob_map * bg_mask  # high predicted prob but actually background
    flat = scores.view(-1)
    K = flat.shape[0]
    if n_hard <= 0:
        return []
    # get top-k indices
    n_hard = min(n_hard, K)
    vals, idx = torch.topk(flat, k=n_hard, largest=True)
    # convert idx to (y,x)
    H, W = pred_prob_map.shape
    ys = (idx // W).cpu().numpy().tolist()
    xs = (idx % W).cpu().numpy().tolist()
    coords = list(zip(ys, xs))
    return coords


def frequency_contrastive_loss(
    freq_embed,
    gt_mask,
    pred_mask_logits=None,
    n_anchors_per_image=64,
    patch_size=7,
    n_negatives=128,
    hard_negative_ratio=0.5,
    temperature=0.07,
    device="cuda",
):
    """
    freq_embed: (B, d, H_e, W_e) -- assumed L2-normalized per-vector in channel dim (or we will renorm after pooling)
    gt_mask: (B,1,H_full,W_full) ground truth binary mask (float 0/1)
    pred_mask_logits: (B,1,H_full,W_full) logits used to select hard negatives; if None, we approximate using gt (or skip hard neg)
    Returns scalar loss (mean over anchors used)
    Design:
      - For each image i, sample up to n_anchors_per_image anchor positions from ground-truth foreground on embedding spatial grid (we resize GT to embedding grid)
      - For each anchor, sample one positive (another pos in same image) and n_neg negatives (mix hard+random)
      - Compute InfoNCE with one positive per anchor (multi-positive variant can be implemented later)
    """
    B, d, He, We = freq_embed.shape
    device = freq_embed.device
    # resize gt_mask & pred_mask to embedding size
    gt_small = F.interpolate(gt_mask, size=(He, We), mode="nearest")  # (B,1,He,We)
    if pred_mask_logits is not None:
        pred_prob = torch.sigmoid(pred_mask_logits)
        pred_small = F.interpolate(
            pred_prob, size=(He, We), mode="bilinear", align_corners=False
        ).squeeze(1)  # (B,He,We)
    else:
        pred_small = None

    anchors = []  # list of (img_idx, y, x)
    positives = []  # list of coords for positive for each anchor (img_idx, y, x)
    negatives = []  # list of list coords for each anchor
    # For vectorized computation we'll collect embeddings per anchor and their positives & negatives
    for b in range(B):
        gt_b = gt_small[b, 0]  # (He,We)
        # sample anchors positions
        pos_coords = torch.nonzero(gt_b, as_tuple=False)  # (K,2)
        K = pos_coords.shape[0]
        if K == 0:
            continue
        n_take = min(n_anchors_per_image, K)
        perm = torch.randperm(K, device=device)[:n_take]
        chosen = pos_coords[perm]  # (n_take,2)
        # for each chosen, sample a positive (another pos)
        for idx in range(chosen.shape[0]):
            yx = chosen[idx]
            y, x = int(yx[0].item()), int(yx[1].item())
            # possible positives are pos_coords excluding this index
            if K <= 1:
                # cannot form positive; skip this anchor
                continue
            # sample positive index not equal to current
            # create indices list
            other_indices = torch.arange(K, device=device)
            # map perm idx to original - need to get global index of chosen
            global_idx = perm[idx]
            mask_others = other_indices != global_idx
            others = other_indices[mask_others]
            pos_choice = others[torch.randint(0, others.shape[0], (1,), device=device)][
                0
            ]
            py, px = (
                int(pos_coords[pos_choice, 0].item()),
                int(pos_coords[pos_choice, 1].item()),
            )
            # negatives: choose n_negatives coords from background
            neg_coords = []
            n_hard = int(n_negatives * hard_negative_ratio)
            n_rand = n_negatives - n_hard
            # hard negatives by pred_small high prob if available
            if (pred_small is not None) and (n_hard > 0):
                hard_coords = sample_hard_negatives(pred_small[b], gt_b, n_hard)
                # note sample_hard_negatives returns list of (y,x)
                neg_coords.extend(hard_coords)
            # fill random background samples
            # get all background positions
            bg_coords = torch.nonzero((gt_b == 0), as_tuple=False)
            Bg = bg_coords.shape[0]
            if Bg == 0:
                # no background -> skip
                continue
            n_rand = min(n_rand, Bg)
            perm2 = torch.randperm(Bg, device=device)[:n_rand]
            rand_coords = bg_coords[perm2]
            for r in rand_coords:
                neg_coords.append((int(r[0].item()), int(r[1].item())))
            # if not enough negatives, pad with random bg repeated
            if len(neg_coords) < n_negatives:
                # repeat random picks
                while len(neg_coords) < n_negatives:
                    r = bg_coords[torch.randint(0, Bg, (1,), device=device)][0]
                    neg_coords.append((int(r[0].item()), int(r[1].item())))
            # store
            anchors.append((b, y, x))
            positives.append((b, py, px))
            negatives.append(neg_coords[:n_negatives])

    if len(anchors) == 0:
        # no anchors => zero loss
        return torch.tensor(0.0, device=device, requires_grad=True)

    # gather embeddings for anchors, positives and negatives
    def get_patch_embedding_at(freq_embed, coord, patch_size):
        # freq_embed: (B,d,He,We) ; coord: (b,y,x)
        b, y, x = coord
        ks = patch_size
        # compute window bounds (clip)
        y0 = max(0, y - ks // 2)
        y1 = min(He, y0 + ks)
        y0 = max(0, y1 - ks)
        x0 = max(0, x - ks // 2)
        x1 = min(We, x0 + ks)
        x0 = max(0, x1 - ks)
        patch = freq_embed[b : b + 1, :, y0:y1, x0:x1]  # (1,d,kh,kw)
        # average pool
        if patch.shape[2] == 0 or patch.shape[3] == 0:
            # degenerate -> fallback to single pixel
            v = freq_embed[b : b + 1, :, y, x]
            v = v.view(1, -1)
        else:
            v = patch.mean(dim=(2, 3))  # (1,d)
        # normalize
        v = F.normalize(v, p=2, dim=1)
        return v  # (1,d)

    device = freq_embed.device
    anchors_emb = []
    pos_emb = []
    neg_embs = []  # list of (n_negatives, d) per anchor
    for i in range(len(anchors)):
        a = anchors[i]
        p = positives[i]
        negs = negatives[i]
        a_e = get_patch_embedding_at(freq_embed, a, patch_size).to(device)  # (1,d)
        p_e = get_patch_embedding_at(freq_embed, p, patch_size).to(device)
        n_e = []
        for coord in negs:
            ne = get_patch_embedding_at(freq_embed, coord, patch_size).to(device)
            n_e.append(ne)
        n_e = torch.cat(n_e, dim=0)  # (n_negatives, d)
        anchors_emb.append(a_e)
        pos_emb.append(p_e)
        neg_embs.append(n_e)

    anchors_emb = torch.cat(anchors_emb, dim=0)  # (N, d)
    pos_emb = torch.cat(pos_emb, dim=0)  # (N, d)
    # Build negatives matrix (N, n_neg, d)
    nN = len(neg_embs)
    n_neg = neg_embs[0].shape[0]
    neg_stack = torch.stack([ne for ne in neg_embs], dim=0)  # (N, n_neg, d)

    # compute similarities
    # numerator: exp(sim(a, pos)/tau)
    sim_pos = torch.sum(anchors_emb * pos_emb, dim=1) / (temperature)  # (N,)
    num = torch.exp(sim_pos)
    # denominator: sum over pos + negatives
    # sim with negatives
    anchors_exp = anchors_emb.unsqueeze(1)  # (N,1,d)
    sim_negs = torch.sum(anchors_exp * neg_stack, dim=2) / (temperature)  # (N, n_neg)
    denom = num + torch.sum(torch.exp(sim_negs), dim=1)
    # loss per anchor:
    loss_per_anchor = -torch.log((num + 1e-10) / (denom + 1e-10))
    loss = loss_per_anchor.mean()
    return loss


def cod_loss(outputs, gt_mask, cfg=None):
    """
    outputs: dict with keys 'masks' (list of 3 logits: coarse->fine), 'edge' (B,1,h_e,w_e), 'freq_embed' (B,d,h_e,w_e)
    gt_mask: (B,1,H,W) binary {0,1} float
    cfg: dict of hyperparameters (weights...). If None, defaults used.
    Returns (loss, dict_losses)
    """

    if cfg is None:
        cfg = {}
    # weights and hyperparams
    alpha_scales = cfg.get(
        "alpha_scales", [0.3, 0.3, 1.0]
    )  # weights for coarse->fine masks
    gamma_dice = cfg.get("gamma_dice", 1.0)
    lambda_b = cfg.get("lambda_b", 1.0)
    lambda_f = cfg.get("lambda_f", 0.2)
    # n_anchors = cfg.get("n_anchors", 64)
    # patch_size = cfg.get("patch_size", 7)
    # n_negatives = cfg.get("n_negatives", 128)
    # hard_neg_ratio = cfg.get("hard_neg_ratio", 0.5)
    # temperature = cfg.get("temperature", 0.07)
    # device = gt_mask.device

    # mask losses (multi-scale)
    masks = outputs["masks"]  # list (coarse->fine)
    assert len(masks) == len(alpha_scales)
    mask_losses = []
    B = gt_mask.shape[0]
    for i, mlogit in enumerate(masks):
        # resize gt to this logit's spatial
        _, _, h, w = mlogit.shape
        gt_small = F.interpolate(gt_mask, size=(h, w), mode="nearest")
        loss_md = bce_dice_loss(
            mlogit, gt_small, bce_weight=1.0, dice_weight=gamma_dice
        )
        mask_losses.append(alpha_scales[i] * loss_md)
    mask_loss = sum(mask_losses)

    # boundary loss
    edge_logits = outputs["edge"]  # (B,1,He,We)
    He, We = edge_logits.shape[2], edge_logits.shape[3]
    gt_edge = make_boundary_target(gt_mask, dilation_radius=1)
    gt_edge_small = F.interpolate(gt_edge, size=(He, We), mode="nearest")
    boundary_loss = bce_loss_fn(edge_logits, gt_edge_small)

    # # frequency contrastive loss
    # freq_embed = outputs["freq_embed"]  # (B,d,He,We) assumed normalized already by FEH
    # # pred_mask for hard negatives: use final fine mask probability if exists
    # final_logits = masks[-1]
    # # upsample final logits to embedding size
    # pred_for_hn = F.interpolate(
    #     final_logits,
    #     size=(freq_embed.shape[2], freq_embed.shape[3]),
    #     mode="bilinear",
    #     align_corners=False,
    # )
    # freq_loss = frequency_contrastive_loss(
    #     freq_embed,
    #     gt_mask,
    #     pred_mask_logits=pred_for_hn,
    #     n_anchors_per_image=n_anchors,
    #     patch_size=patch_size,
    #     n_negatives=n_negatives,
    #     hard_negative_ratio=hard_neg_ratio,
    #     temperature=temperature,
    #     device=device,
    # )

    # total loss
    total_loss = mask_loss + lambda_b * boundary_loss + lambda_f  # * freq_loss

    loss_dict = {
        "mask": mask_loss,
        "boundary": boundary_loss,
        "total": total_loss,
        # "freq_loss": freq_loss.detach(),
    }
    return loss_dict
