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


class LossFactory:
    @staticmethod
    def get_loss(config: Config):
        if config.loss == 'clip':
            return CLIPLoss()
        else:
            raise NotImplemented
