"""CPU smoke test — verify model forward / MDLM loss / sampling / prompt anchoring
before renting a GPU.  Run:  python smoke_test.py
Uses a tiny config so it runs in seconds on CPU."""
import torch
import torch.nn.functional as F
from model import DiffusionGPT, ModelArgs


def main():
    a = ModelArgs(vocab_size=256, block_size=32, n_layer=2, n_head=4, n_embd=64,
                  num_steps=8, device="cpu")
    m = DiffusionGPT(a)
    print("params:", sum(p.numel() for p in m.parameters()))
    print("mask_id:", m.mask_token_id, "n_out:", m.n_out)

    # one training step (mirrors train.compute_masked_loss)
    B, T = 4, a.block_size
    x0 = torch.randint(0, a.vocab_size, (B, T))
    t = torch.rand(B).clamp(min=1e-3)
    mask = torch.rand(B, T) < t.unsqueeze(1)
    xt = torch.where(mask, m.mask_token_id, x0)
    logits = m(xt, t)
    assert logits.shape == (B, T, a.vocab_size + 1), logits.shape
    ce = F.cross_entropy(logits.view(-1, a.vocab_size + 1), x0.view(-1),
                         reduction="none").view(B, T)
    loss = ((1.0 / t) * (ce * mask.float()).sum(1) / T).mean()
    loss.backward()
    gnorm = sum(p.grad.norm() ** 2 for p in m.parameters() if p.grad is not None) ** 0.5
    print(f"train step OK | loss={loss.item():.3f} | grad_norm={gnorm.item():.3f}")

    # sampling must resolve every mask and never emit the MASK id
    frames = list(m.sample_stream(batch_size=2, seq_len=T, num_steps=a.num_steps,
                                  temperature=0.9, top_k=20))
    final = frames[-1]["x_t"]
    assert not (final == m.mask_token_id).any(), "unresolved MASK in output"
    assert final.max().item() < a.vocab_size, "emitted MASK/out-of-range id"
    print(f"stream frames={len(frames)} | fully resolved | max_id={final.max().item()}")

    # prompt conditioning: anchors stay fixed
    prompt = [1, 2, 3, 4, 5]
    f2 = list(m.sample_stream(batch_size=1, seq_len=T, num_steps=a.num_steps, prompt_ids=prompt))
    kept = f2[-1]["x_t"][0, :5].tolist()
    assert kept == prompt, f"anchors changed: {kept}"
    print("prompt anchors preserved:", kept)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
