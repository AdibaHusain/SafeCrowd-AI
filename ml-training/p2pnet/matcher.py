import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class P2PLoss(nn.Module):
    """
    For each image:
      1. Match predicted points to ground-truth points via the Hungarian
         algorithm (cost = pairwise L2 distance) — no NMS or fixed boxes
         needed.
      2. Matched anchors get label=1 (person) + a regression target (the GT
         point they were matched to). All other anchors get label=0
         (background) and no regression loss.
    """
    def __init__(self, cls_weight: float = 1.0, reg_weight: float = 0.0002):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.ce = nn.CrossEntropyLoss()
        self.smooth_l1 = nn.SmoothL1Loss(reduction="sum")

    def forward(self, pred_points, cls_logits, gt_points_list):
        """
        pred_points: (B, N, 2)
        cls_logits:  (B, N, 2)
        gt_points_list: list of length B, each a (M_i, 2) tensor of GT points
        """
        batch_size = pred_points.shape[0]
        total_cls_loss = 0.0
        total_reg_loss = torch.tensor(0.0, device=pred_points.device)
        total_matched = 0

        for b in range(batch_size):
            gt_points = gt_points_list[b]
            preds = pred_points[b]
            logits = cls_logits[b]

            labels = torch.zeros(preds.shape[0], dtype=torch.long, device=preds.device)

            if gt_points.shape[0] > 0:
                cost = torch.cdist(preds, gt_points, p=2).detach().cpu().numpy()
                pred_idx, gt_idx = linear_sum_assignment(cost)

                labels[pred_idx] = 1
                total_reg_loss = total_reg_loss + self.smooth_l1(preds[pred_idx], gt_points[gt_idx])
                total_matched += len(pred_idx)

            total_cls_loss = total_cls_loss + self.ce(logits, labels)

        cls_loss = total_cls_loss / batch_size
        reg_loss = total_reg_loss / max(total_matched, 1)

        loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        return loss, cls_loss, reg_loss