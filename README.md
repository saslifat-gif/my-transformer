# my-transformer

A decoder-only transformer trained on 三体 (The Three-Body Problem), built from scratch in PyTorch. Inspired by Karpathy's nanoGPT and the "Attention Is All You Need" paper.

## Architecture

- Decoder-only transformer (GPT-style)
- Multi-head self-attention with RoPE positional encoding
- BPE tokenizer trained from scratch on the corpus
- Pre-norm residual blocks (LayerNorm before attention and FFN)

## Hyperparameters

```text
vocab_size = 8000
block_size = 64
batch_size = 128
n_embd    = 128
n_head    = 4
n_layers  = 3
params    = 2.65M
```

## Files

- `My_trans_v1.py` — Mac/MPS version with LayerNorm
- `My_trans_v1_win.py` — Windows/CUDA version, adds RMSNorm and StochasticRMSNorm experiments
- `BPE_training.py` — standalone BPE tokenizer, saves vocab as `ids8000.json` and `merges8000.json`
- `santi.txt` — training corpus (~900K chars raw, 5.07x compression after BPE)

## BPE Tokenizer

Trained from scratch on 三体. UTF-8 byte-level, 8000 vocab size.

```text
raw length:  2,670,321 chars
encoded:       526,372 tokens
compression:        5.07x
unique chars:       3,648
```

## Normalization Experiment — StochasticRMSNorm

Tested three normalization methods at matched total dropout (0.2):

| Config | internal p | external dropout | total |
|---|---|---|---|
| LayerNorm | - | 0.2 | 0.2 |
| StochasticRMSNorm | 0.1 | 0.1 | 0.2 |
| RMSNorm | - | 0.2 | 0.2 |

**Convergence speed (steps → train loss):**
<img width="2940" height="1666" alt="step_vs_train_loss_convergence" src="https://github.com/user-attachments/assets/6923a921-9ecb-41e0-b44f-64ce5e92f4a4" />

**Generalization (train loss → val loss):**
<img width="2880" height="1536" alt="train_vs_val_loss_norm_comparison" src="https://github.com/user-attachments/assets/10937308-8836-42c4-8b8a-8306ca94f74b" />
**Findings:** StochasticRMSNorm converges fastest across all steps. Generalization performance is similar across all three configs — the val loss curves are close when compared at matched train loss. Whether the convergence advantage comes from the internal masking mechanism or the different distribution of regularization (internal vs external) remains unresolved. `eps=1e-8` is identical across all implementations, ruling it out as a confound.

**note:** StochasticRMSNorm shows faster training convergence, but whether this reflects a genuine optimization advantage or an artifact of the internal/external regularization split remains unconfirmed.

**StochasticRMSNorm:**
```python
class StochasticRMSNorm(nn.Module):
    """
    RMSNorm with stochastic feature masking during training.
    Randomly masks features before computing RMS scale estimate.
    Falls back to standard RMSNorm at inference.
    Args:
        d (int): model dimension
        p (float): mask probability, default 0.1
        eps (float): stability term, default 1e-8
    """
    def __init__(self, d, p=0.1, eps=1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))
        self.p = p
        self.eps = eps

    def forward(self, x):
        if self.training:
            mask = torch.bernoulli(torch.ones_like(x) * (1 - self.p))
            rms = (x * mask).pow(2).sum(-1, keepdim=True)
            rms = rms / mask.sum(-1, keepdim=True).clamp(min=1)
            rms = rms.sqrt()
        else:
            rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        return self.gamma * x / (rms + self.eps)
```
