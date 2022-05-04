import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(self, epsilon: float = 0.1, alpha = None, gamma = 2., reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.nll_loss = torch.nn.NLLLoss(
            weight=alpha, reduction='none')

    def forward(self, preds, target):
        n = preds.size()[-1]

        log_p = F.log_softmax(preds, dim=-1)
        ce = self.nll_loss(log_p, target)
        # label smoothing
        ce = self.epsilon * -log_p.sum(dim=-1)/ n + (1 - self.epsilon) * ce

        # get true class column from each row
        all_rows = torch.arange(len(preds))
        log_pt = log_p[all_rows, target]
        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma
        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


input = torch.randn(3, 5, requires_grad=True)
target = torch.randint(5, (3,), dtype=torch.int64)

loss = FocalLoss(reduction='sum')
print(loss(input,target))
