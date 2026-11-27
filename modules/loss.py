import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from config.base_config import Config

class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims, logit_scale):
        """
        Inputs: cosine similarities
            sims: n x n (text is dim-0)
            logit_scale: 1 x 1
        """
        logit_scale = logit_scale.exp()
        logits = sims * logit_scale
        
        t2v_log_sm = F.log_softmax(logits, dim=1)
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()

        v2t_log_sm = F.log_softmax(logits, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()

        return (t2v_loss + v2t_loss) / 2.0


class MarginCLIPLoss(CLIPLoss):
    """Symmetric CLIP loss with a fixed additive cosine margin on positives.

    Training only: retrieval still ranks the unmodified cosine similarities.
    A zero margin delegates to the original implementation exactly.
    """
    def __init__(self, margin=0.02):
        super().__init__()
        if not math.isfinite(margin) or not 0 <= margin < 1:
            raise ValueError('margin must be finite and in [0, 1)')
        self.margin = float(margin)

    def forward(self, sims, logit_scale):
        if sims.ndim != 2 or sims.shape[0] != sims.shape[1] or sims.shape[0] == 0:
            raise ValueError('One positive per row/column requires a nonempty square matrix')
        if self.margin == 0:
            return super().forward(sims, logit_scale)
        diagonal = torch.eye(sims.shape[0], device=sims.device, dtype=sims.dtype)
        return super().forward(sims - self.margin * diagonal, logit_scale)


class SupConCLIPLoss(CLIPLoss):
    """Cross-modal adaptation of SupCon (Khosla et al., NeurIPS 2020, Eq. 2).

    Equal video IDs define positives, including the original paired diagonal.
    Average log-probabilities over positives outside the log, in both directions.
    No extra encoder, projection head, temperature or inference change.
    """
    def forward(self, sims, logit_scale, video_ids=None):
        if sims.ndim != 2 or sims.shape[0] != sims.shape[1] or sims.shape[0] == 0:
            raise ValueError('SupCon requires a nonempty square paired similarity matrix')
        if video_ids is None:
            raise ValueError('SupCon requires video_id for every pair')
        if isinstance(video_ids, torch.Tensor):
            if video_ids.ndim != 1:
                raise ValueError('video_ids must be one-dimensional')
            video_ids = video_ids.detach().cpu().tolist()
        if isinstance(video_ids, (str, bytes)) or len(video_ids) != sims.shape[0]:
            raise ValueError('SupCon requires one video_id per paired row/column')
        if any(value is None for value in video_ids):
            raise ValueError('Missing video_id in SupCon batch')

        # Support string MSR-VTT IDs and numeric IDs without hashing collisions.
        mapping = {}
        encoded = [mapping.setdefault(value, len(mapping)) for value in video_ids]
        if len(mapping) == len(encoded):
            # Exact legacy loss and gradients if there are no extra positives.
            return super().forward(sims, logit_scale)

        ids = torch.tensor(encoded, device=sims.device, dtype=torch.long)
        logits = sims * logit_scale.exp()
        mask = (ids[:, None] == ids[None, :]).to(dtype=logits.dtype)
        target_t2v = mask / mask.sum(dim=1, keepdim=True)
        target_v2t = mask / mask.sum(dim=0, keepdim=True)
        t2v = -(target_t2v * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        v2t = -(target_v2t * F.log_softmax(logits, dim=0)).sum(dim=0).mean()
        return (t2v + v2t) / 2.0


def paired_loss(loss_fn, sims, logit_scale, data):
    """Pass metadata only to SupCon; preserve existing two-argument losses."""
    if isinstance(loss_fn, SupConCLIPLoss):
        return loss_fn(sims, logit_scale, video_ids=data.get('video_id'))
    return loss_fn(sims, logit_scale)


class LossFactory:
    @staticmethod
    def get_loss(config: Config):
        if config.loss == 'clip':
            return CLIPLoss()
        elif config.loss == 'supcon':
            return SupConCLIPLoss()
        else:
            raise ValueError(f'Unknown loss: {config.loss}')
