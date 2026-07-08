import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelArgs:
    # --- Nepali masked-diffusion config (~50M params) ---
    # vocab_size = number of REAL tokens from the Nepali SentencePiece BPE.
    # The absorbing [MASK] token lives at id == vocab_size, so the embedding
    # table / lm_head span vocab_size + 1 rows (see DiffusionGPT.__init__).
    vocab_size: int = 16384    # Nepali SentencePiece BPE (was 50304 GPT-2 BPE)
    block_size: int = 512      # shorter blocks: faster denoise, cleaner viz, Nepali paragraphs fit
    n_layer: int = 12          # keep depth
    n_head: int = 8            # 512 / 8 = 64 head dim (Tensor-Core aligned)
    n_embd: int = 512          # shrunk from 768 -> 512 (~50M total params)
    dropout: float = 0.1
    num_steps: int = 64        # DEFAULT number of inference denoising steps (not used at train time)
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding of the CONTINUOUS diffusion time t in (0, 1]."""
    def __init__(self, n_embd, scale: float = 1000.0):
        super().__init__()
        self.n_embd = n_embd
        self.scale = scale  # spread continuous t in (0,1] across the sinusoid frequency band
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.SiLU(),
            nn.Linear(n_embd, n_embd)
        )

    def forward(self, t):
        # t: float tensor of shape (B,), values in (0, 1]
        t = t.float() * self.scale
        half_dim = self.n_embd // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        return self.mlp(emb)

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.n_embd % args.n_head == 0
        self.n_head = args.n_head
        self.head_dim = args.n_embd // args.n_head
        self.wq = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.wk = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.wv = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.wo = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout_p = args.dropout

    def forward(self, x, freqs_cis):
        B, T, C = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # FlashAttention-2 enabled automatically via SDPA in PyTorch 2.x
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False  # Bidirectional attention for diffusion (full-sequence denoising)
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.wo(y))

class Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ln1 = RMSNorm(args.n_embd)
        self.attn = SelfAttention(args)
        self.ln2 = RMSNorm(args.n_embd)
        hidden_dim = int(8 * args.n_embd / 3)
        hidden_dim = 256 * ((hidden_dim + 255) // 256)  # Aligned to 256
        self.ffn = nn.Sequential(
            SwiGLU(args.n_embd, hidden_dim),
            nn.Dropout(args.dropout)
        )

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.ln1(x), freqs_cis)
        x = x + self.ffn(self.ln2(x))
        return x

def cosine_mask_frac(frac):
    """Masked fraction at normalized diffusion position frac in [0, 1].
    frac=1 -> fully masked (1.0), frac=0 -> fully clean (0.0). MaskGIT cosine reveal."""
    return math.cos((1.0 - frac) * (math.pi / 2.0))

class DiffusionGPT(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        # Dedicated absorbing [MASK] token at id == vocab_size (cleaner than stealing an in-vocab id).
        # Real tokens: 0 .. vocab_size-1.  MASK: vocab_size.  Table spans vocab_size + 1.
        self.mask_token_id = args.vocab_size
        self.n_out = args.vocab_size + 1

        self.token_embedding = nn.Embedding(self.n_out, args.n_embd)
        self.time_embedding = TimestepEmbedding(args.n_embd)

        freqs_cis = precompute_freqs_cis(args.n_embd // args.n_head, args.block_size)
        self.register_buffer("freqs_cis", freqs_cis)

        self.blocks = nn.ModuleList([Block(args) for _ in range(args.n_layer)])
        self.ln_f = RMSNorm(args.n_embd)
        self.lm_head = nn.Linear(args.n_embd, self.n_out, bias=False)

        # Weight tying (halves the dominant param cost at this scale)
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, x, t):
        # x: (B, T) token ids incl. MASK.  t: (B,) continuous time in (0, 1].
        B, T = x.shape
        x = self.token_embedding(x) + self.time_embedding(t).unsqueeze(1)
        freqs_cis = self.freqs_cis[:T]

        for block in self.blocks:
            x = block(x, freqs_cis)

        return self.lm_head(self.ln_f(x))

    def get_mask_prob(self, frac):
        """Cosine masked fraction for a normalized position frac in [0, 1] (inference reveal schedule)."""
        return cosine_mask_frac(float(frac))

    @staticmethod
    def _filter_logits(logits, top_k=None, top_p=None):
        """Apply top-k then top-p (nucleus) filtering along the vocab dim (last)."""
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < v[..., [-1]], -float('Inf'))
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum > top_p
            remove[..., 1:] = remove[..., :-1].clone()  # keep the first token that crosses p
            remove[..., 0] = False
            remove = torch.zeros_like(remove).scatter(-1, sorted_idx, remove)
            logits = logits.masked_fill(remove, -float('Inf'))
        return logits

    # -----------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------
    @torch.no_grad()
    def sample_stream(self, batch_size=1, seq_len=None, num_steps=None,
                      temperature=1.0, top_k=None, top_p=None, remask_noise=0.1,
                      device=None, prompt_ids=None):
        """Iterative confidence-based (MaskGIT) unmasking. Yields the state after
        EACH denoising step so callers (TUI / web UI) can animate the reveal.

        Each yielded dict:
            x_t          : (B, seq_len) current token ids (MASK where unresolved)
            newly        : (B, seq_len) bool, positions decoded THIS step
            anchor       : (B, seq_len) bool, fixed prompt positions
            confidence   : (B, seq_len) float per-token confidence of the current guess
            step, total  : ints (step counts DOWN from total to 1)
        """
        self.eval()
        device = device or self.args.device
        seq_len = seq_len or self.args.block_size
        num_steps = num_steps or self.args.num_steps

        x_t = torch.full((batch_size, seq_len), self.mask_token_id, dtype=torch.long, device=device)
        anchor = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        if prompt_ids is not None and len(prompt_ids) > 0:
            p = torch.tensor(prompt_ids[:seq_len], dtype=torch.long, device=device)
            x_t[:, :p.numel()] = p
            anchor[:, :p.numel()] = True

        for t_val in reversed(range(1, num_steps + 1)):
            frac = t_val / num_steps
            t = torch.full((batch_size,), frac, dtype=torch.float32, device=device)
            logits = self.forward(x_t, t) / temperature
            logits[..., self.mask_token_id] = -float('Inf')  # never emit the MASK token
            logits = self._filter_logits(logits, top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs.view(-1, probs.size(-1)))
            x_0_pred = dist.sample().view(batch_size, seq_len)

            currently_masked = (x_t == self.mask_token_id)
            chosen_probs = torch.gather(probs, 2, x_0_pred.unsqueeze(-1)).squeeze(-1)

            # Gumbel-noised confidence, MaskGIT-style, only over still-masked slots.
            # Higher remask_noise randomizes the unmask order -> breaks the
            # high-frequency-token-first cascade that drives repetition.
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(chosen_probs) + 1e-9) + 1e-9)
            confidence = chosen_probs + (remask_noise * gumbel_noise)
            confidence = confidence.masked_fill(~currently_masked, -float('inf'))

            # How many to KEEP masked after this step (cosine reveal on the next position).
            # Force 0 on the final step so nothing is left masked (ceil(~0) would round up to 1).
            next_mask_prob = cosine_mask_frac((t_val - 1) / num_steps)
            num_to_keep_masked = 0 if t_val == 1 else int(math.ceil(next_mask_prob * seq_len))
            num_masked_per_row = currently_masked.sum(dim=1)
            num_to_unmask = (num_masked_per_row - num_to_keep_masked).clamp(min=0)

            order = torch.argsort(confidence, dim=1, descending=True)
            rank = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
            unmask_this_step = rank < num_to_unmask.unsqueeze(1)
            unmask_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            unmask_positions.scatter_(1, order, unmask_this_step)

            x_next = torch.where(unmask_positions, x_0_pred, x_t)
            newly = currently_masked & (x_next != self.mask_token_id)
            x_t = x_next

            # Report the model's best current guess everywhere (for confidence coloring in the viz)
            best_conf = chosen_probs.masked_fill(~currently_masked & ~newly, 1.0)

            yield {
                'x_t': x_t.clone(),
                'newly': newly.clone(),
                'anchor': anchor.clone(),
                'confidence': best_conf.clone(),
                'step': t_val,
                'total': num_steps,
            }

        self.train()

    @torch.no_grad()
    def sample(self, batch_size, seq_len, temperature=1.0, top_k=None, top_p=None,
               remask_noise=0.1, device=None, num_steps=None, prompt_ids=None):
        """Convenience wrapper: run the full denoise and return only the final tokens."""
        x_t = None
        for frame in self.sample_stream(batch_size=batch_size, seq_len=seq_len,
                                        num_steps=num_steps, temperature=temperature,
                                        top_k=top_k, top_p=top_p, remask_noise=remask_noise,
                                        device=device, prompt_ids=prompt_ids):
            x_t = frame['x_t']
        return x_t
