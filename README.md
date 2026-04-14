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

Explored a custom norm layer that applies stochastic feature masking during the RMS computation, making the scale estimate itself noisy during training.

| Config | internal p | external dropout | total dropout | Val best | Val @2800 |
|---|---|---|---|---|---|
| LayerNorm | - | 0.2 | 0.2 | 7.22 @800 | 7.95 |
| StochasticRMS | 0.1 | 0.1 | 0.2 | 7.20 @600 | 8.99 |
| RMSNorm | - | 0.2 | 0.2 | 7.21 @600 | 9.01 |
| StochasticRMS | 0.3 | 0.1 | 0.4 | 7.19 @600 | 8.93 |
| RMSNorm | - | 0.4 | 0.4 | 7.18 @600 | 9.02 |

**Findings:** LayerNorm dominates overall. At higher dropout, StochasticRMSNorm begins to outperform standard RMSNorm — the internal masking adds regularization value beyond what external dropout alone provides. All configs overfit after ~600-800 steps, consistent with small-data regime.