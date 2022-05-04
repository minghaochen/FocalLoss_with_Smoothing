# Multi-class Focal Loss with label smoothing

An implementation of multi-class Focal Loss with labeling smoothing.

- focal loss

$$
CE(p_t)=-\log (p_t)\\
FL(p_t)=-(1-p_t)^{\gamma}\log (p_t)
$$

- label smoothing

```
before:if i == y: p_i = 1, else p_i = 0
after: if i == y: p_i = (1-\epsilon), else p_i = \epsilon / (n_class)
```

## Usage

```
loss_fun = FocalLoss(reduction='sum')
...
input = torch.randn(3, 5, requires_grad=True)
target = torch.randint(5, (3,), dtype=torch.int64)
loss = loss_fun(input,target)
```

