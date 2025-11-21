import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """
    Compound Loss: Reconstruction + Consistency + Alignment + Separation
    """

    def __init__(self, lambda_rec=1.0, lambda_con=1.0, lambda_align=0.5, lambda_sep=2.0, margin=0.5):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_con = lambda_con
        self.lambda_align = lambda_align
        self.lambda_sep = lambda_sep
        self.margin = margin
        self.mse = nn.MSELoss()

    def triplet_loss(self, embeddings, labels):
        # Simple Batch Hard Triplet Loss implementation
        dist_mat = torch.cdist(embeddings, embeddings)
        loss = 0.0
        bs = embeddings.size(0)
        valid_triplets = 0

        for i in range(bs):
            anchor = embeddings[i]
            pos_mask = labels == labels[i]
            neg_mask = labels != labels[i]

            pos_mask[i] = False  # Exclude self

            if not pos_mask.any() or not neg_mask.any():
                continue

            pos_dists = dist_mat[i][pos_mask]
            neg_dists = dist_mat[i][neg_mask]

            hardest_pos = pos_dists.max()
            hardest_neg = neg_dists.min()

            curr_loss = F.relu(hardest_pos - hardest_neg + self.margin)
            loss += curr_loss
            valid_triplets += 1

        return loss / (valid_triplets + 1e-8)

    def forward(self, out_a, out_b, target, feat_a, feat_b, labels):
        # 1. Reconstruction Loss (Student reconstructs Target)
        loss_rec = self.mse(out_b, target)

        # 2. Consistency Loss (Teacher vs Student Output)
        loss_con = self.mse(out_a, out_b)

        # 3. Alignment Loss (Feature Cosine Similarity)
        fa_flat = feat_a.view(feat_a.size(0), -1)
        fb_flat = feat_b.view(feat_b.size(0), -1)
        cosine_sim = F.cosine_similarity(fa_flat, fb_flat).mean()
        loss_align = 1.0 - cosine_sim

        # 4. Separation Loss (Triplet Loss on Student Features)
        fb_gap = F.adaptive_avg_pool2d(feat_b, (1, 1)).view(feat_b.size(0), -1)
        loss_sep = self.triplet_loss(fb_gap, labels)

        total = (self.lambda_rec * loss_rec +
                 self.lambda_con * loss_con +
                 self.lambda_align * loss_align +
                 self.lambda_sep * loss_sep)

        return total, {
            "rec": loss_rec.item(),
            "con": loss_con.item(),
            "align": loss_align.item(),
            "sep": loss_sep.item()
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification
    """

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()