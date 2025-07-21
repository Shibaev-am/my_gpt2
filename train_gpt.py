import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class LayerNorm(nn.Module):
    def __init__(self, emd_size, bias):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(emd_size))
        self.bias = nn.Parameter(torch.zeros(emd_size)) if bias else None

    def forward(self, inputs):
        # inputs: (b, t, emb_size)
        return F.layer_norm(inputs, self.weight.shape, self.weight, self.bias, 1e-5)


class QKV_Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.emb_size % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_size = config.emb_size // config.n_heads

        self.proj = nn.Linear(config.emb_size, 3 * self.head_size * self.n_heads)
        self.final_proj = nn.Linear(self.head_size * self.n_heads, config.emb_size)
        self.register_buffer("mask",
                             torch.tril(torch.ones(config.context_size, config.context_size))
                             .view(1, 1, config.context_size, config.context_size))

        self.attn_dropout = nn.Dropout(config.dropout)
        self.final_dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        B, T, C = inputs.size()

        q, k, v = self.proj(inputs).split(self.n_heads * self.head_size, dim=2)
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        attn = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(self.head_size))
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        y = attn @ v

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_size)
        y = self.final_dropout(self.final_proj(y))

        return y



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.emb_size, 4 * config.emb_size, config.bias)
        self.activation = nn.GELU('tanh')
        self.fc2 = nn.Linear(4 * config.emb_size, config.emb_size, config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = LayerNorm(config.emb_size, config.bias),
        self.attn = QKV_Attention(config),
        self.ln_2 = LayerNorm(config.emb_size, config.bias),
        self.mlp = MLP(config)


    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_1(x))

        return x




class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.emb_size),
            wpe = nn.Embedding(config.context_size, config.emb_size),
            blocks = nn.ModuleList([Block(config) for _ in range(config.block_count)]),
            lh_f = LayerNorm(config)
        ))

        self.lm_head = nn.Linear(config.emb_size, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights())

    def forward(self, idx, targets):
        device = idx.device
        b, t = idx.size()

        # ids: (b, t)
        token_embs = self.transformer.wte(idx) # (b, t, emb_size)
        position_embs = self.transformer.wpe(torch.arange(0, t, dtype=torch.long, device=device)) # (t, emb_size)

        x = token_embs + position_embs

        for block in self.transformer.blocks:
            x = block(x)


        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.weight)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):

        param_dict = {pn: p for pn, p in self.named_parameters()}

        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx

import tiktoken

class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        self.curr_pos = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.curr_pos : self.curr_pos + T*B+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        self.curr_pos += B*T

        if self.curr_pos + B*T+1 > len(self.tokens):
            self.curr_pos = 0

        return x, y


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'

print(f'device: {device}')

train_loader = DataLoader(B=4, T=32)

model = GPT(GPTConfig())
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(52):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()
    logits, loss = model(x, y)

    loss.backward()
    optimizer.step()
    print(f'step {i}, loss {loss.item()}')
