import torch
from torch import nn
from torch.nn import functional as F
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('/kaggle/working/my-transformer/santi.txt', "r", encoding = 'utf-8') as f:
    text = f.read()
tokens = list(text.encode("utf-8"))
vocab_size = 8000
block_size = 64
batch_size = 128
n_embd = 128
n_head = 4
layer_n = 3
train_loop = 5000
EXTERNAL_DROPOUT = 0.2
STOCHASTIC_P = 0.0
NORM_TYPE = 'layernorm'
head_size = n_embd // n_head
#----------BPE tokenizer----------

# load merges
with open(f'/kaggle/working/my-transformer/merges{vocab_size}.json', 'r') as f:
    merges_serializable = json.load(f)
merges = {int(k): tuple(v) for k, v in merges_serializable.items()}

# rebuild vocab
vocab = {idx: bytes([idx]) for idx in range(256)}
for idx, (p0, p1) in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

# load ids directly, skip BPE training
with open(f'/kaggle/working/my-transformer/ids{vocab_size}.json', 'r') as f:
    ids = json.load(f)

def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and pair[0] == ids[i] and pair[1] ==ids[i + 1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def encode(text):
    ids = list(text.encode('utf-8'))
    for idx, pair in merges.items():
        ids = merge(ids, pair, idx)
    return ids

def decode(ids):
    ids = b"".join([vocab[idx] for idx in ids])
    return ids.decode('utf-8', errors = 'replace')

print(f"raw data length {len(tokens)}")
print(f"new data length {len(ids)}")
print(f"ration = {len(tokens)/len(ids):.2f}x")

#----------Data preparation----------

data = torch.tensor(ids, dtype = torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint((len(data) - block_size - 1), (batch_size, ))
    x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
    return x, y

#----------Components----------

@torch.no_grad()
def estimate_loss(step):
    model.eval()
    results = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(100):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses.append(loss.mean().item())
        results[split] = sum(losses) / len(losses)
    print(f"step {step} | train loss: {results['train']:.4f} | val loss: {results['val']:.4f}")
    model.train()

def rope(x):
    B, T, head_size = x.shape
    position = torch.arange(T, device=device).unsqueeze(1) # T, 1
    dim = torch.arange(0, head_size, 2, device=device) # head_size/2
    theta = (1 / 10000**(dim/head_size)) # head_size/2
    angles = (theta * position) # T, head_size/2

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    x_rope = torch.stack([
        x1 * torch.cos(angles) - x2 * torch.sin(angles) , # B, T, head_size/2, 2
        x1 * torch.sin(angles) + x2 * torch.cos(angles),
    ], dim=-1).flatten(-2) # B, T, head_size/2 * 2

    return x_rope #B, T, head_size

class StochasticRMSNorm(nn.Module):
    """
    RMSNorm with stochastic feature masking during training.
    Randomly masks features before computing RMS scale estimate.
    Falls back to standard RMSNorm at inference.

    Outperforms RMSNorm at p > 0.5 on small datasets.
    Still underperforms LayerNorm (missing mean centering).

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
            mask = torch.bernoulli(
                torch.ones_like(x) * (1 - self.p)
            )
            rms = (x * mask).pow(2).sum(-1, keepdim=True)
            rms = rms / mask.sum(-1, keepdim=True).clamp(min=1)
            rms = rms.sqrt()
        else:
            # standard RMS at inference
            rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        
        return self.gamma * x / (rms + self.eps)

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        return self.gamma * x / (rms + self.eps)

class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.W_q = nn.Linear(n_embd, head_size, bias = False) #take n_embd(how many)
        self.W_k = nn.Linear(n_embd, head_size, bias = False) #to n_head(how distinct)
        self.W_v = nn.Linear(n_embd, head_size, bias = False) #each one has head_size(how large)

    def forward(self, x): # x = B, T, n_embd
        T = x.shape[-2]

        w_q = self.W_q(x) # B, T, head_size
        w_k = self.W_k(x)
        w_v = self.W_v(x)

        w_q = rope(w_q)
        w_k = rope(w_k)

        weight = w_q @ w_k.transpose(-2, -1) # B, T, head_size @ B, head_size, T -> B, T, T
        weight = weight * (head_size**(-0.5)) # prevent softmax collapses 
        tril = torch.tril(torch.ones(T, T, device=x.device))
        weight = weight.masked_fill(tril == 0, float('-inf'))
        weight = torch.softmax(weight, dim=-1)
        out = weight @ w_v # -> B, T, head_size

        return out # B, T, head_size
    
class MultiHead(nn.Module):
    def __init__(self, n_head, head_size, n_embd):
        super().__init__()
        self.att = nn.ModuleList([Head(n_embd, head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_head * head_size, n_embd)
    
    def forward(self, x): # x = B, T, n_embd
        out = torch.cat([h(x) for h in self.att], dim=-1) # B, T,  n_head * head_size
        return self.proj(out) # B, T, n_embd
    
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
        )
    
    def forward(self, x):
        return self.net(x) 
    
class Block(nn.Module):
    def __init__(self, n_head, head_size, n_embd):
        super().__init__()
        self.att = MultiHead(n_head, head_size, n_embd)
        self.ffn = FeedFoward(n_embd)
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.drop = nn.Dropout(p=EXTERNAL_DROPOUT)
        self.rms1 = RMSNorm(n_embd)
        self.rms2 = RMSNorm(n_embd)
        self.storms1 = StochasticRMSNorm(n_embd, p=STOCHASTIC_P)
        self.storms2 = StochasticRMSNorm(n_embd, p=STOCHASTIC_P)

    def get_norm(self, norm, x, position):
        if norm == 'rms':
            return self.rms1(x) if position == 1 else self.rms2(x)
        elif norm == 'stochastic_rms':
            return self.storms1(x) if position == 1 else self.storms2(x)
        elif norm == 'layernorm':
            return self.ln_1(x) if position == 1 else self.ln_2(x)

    def forward(self, x):
        x = x + self.drop(self.att(self.get_norm(NORM_TYPE, x, 1)))
        x = x + self.drop(self.ffn(self.get_norm(NORM_TYPE, x, 2)))
        return x

#----------My_transmodel----------

class My_transmodel(nn.Module):
    def __init__(self, n_head, n_embd, head_size,layer_n):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, n_embd)
        self.block = nn.Sequential(*[Block(n_head, head_size, n_embd) for _ in range(layer_n)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.project_out = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):
        logits = self.embedding_table(idx)
        logits = self.block(logits)
        logits = self.ln_f(logits)
        logits = self.project_out(logits)

        if targets is not None:
            B, T, C =logits.shape
            logits = logits.view(B*T, C) # loss function need N(all the tokens be flattened), C not B, T, C
            targets = targets.view(B*T,)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    
    def generate(self, ids, max_netx_tokens):
        for _ in range(max_netx_tokens):
            idx_cond = ids[: , -block_size:] # B, T
            logits, _ = self(idx_cond)
            logits = logits[: ,-1, :]
            prob = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(prob, num_samples=1)
            ids = torch.cat([ids, idx_next], dim=1)
        return ids

#----------Train----------
model = My_transmodel(n_head, n_embd, head_size, layer_n).to(device)
model = nn.DataParallel(model) # multi gpu train
optimzer = torch.optim.Adam(model.parameters(), lr=1e-3)
total_paremeters = sum(p.numel() for p in model.parameters())
print(f'paremeters:{total_paremeters/1e6}M')

for step in range(train_loop):
    if step % 200 == 0:
        estimate_loss(step)
   
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    loss = loss.mean() # the dataprallet given verctor per gpu not scalar
    optimzer.zero_grad(set_to_none=True)
    loss.backward()
    optimzer.step()
    

#-------Eval-----

text = '你'
ids = encode(text)
ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
ids = model.module.generate(ids, max_netx_tokens=100)
print(decode(ids[0].tolist()))