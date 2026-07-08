"""Model-based scorers: generative perplexity (frozen Nepali AR model) and MAUVE.

gen-PPL:  fluency proxy. Score the DIFFUSION model's samples under a frozen Nepali
          AUTOREGRESSIVE model (Sakonii/distilgpt2-nepali). Lower = more fluent-looking,
          BUT it is gameable by repetition -> always read alongside diversity + the
          reference gen-PPL (human text scored by the same model = the target band).
MAUVE:    distribution-level similarity between generated and human text, using a
          multilingual/Indic encoder (MuRIL) to featurize. ~1.0 = indistinguishable.
"""
import torch


def gen_perplexity(texts, model_name="Sakonii/distilgpt2-nepali", device=None,
                   max_length=256):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    total_nll, total_tok = 0.0, 0
    with torch.no_grad():
        for t in texts:
            if not t.strip():
                continue
            ids = tok(t, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
            if ids.size(1) < 2:
                continue
            out = lm(ids, labels=ids)
            ntok = ids.size(1) - 1                 # next-token targets
            total_nll += out.loss.item() * ntok
            total_tok += ntok
    if total_tok == 0:
        return float("nan")
    return float(torch.exp(torch.tensor(total_nll / total_tok)))


def mauve_score(gen_texts, ref_texts, device=None, max_text_length=256,
                featurizers=("google/muril-base-cased", "xlm-roberta-base")):
    try:
        import mauve  # noqa
    except Exception as e:
        return {"mauve": None, "error": f"mauve-text not installed: {e}"}
    device_id = 0 if (device or ("cuda" if torch.cuda.is_available() else "cpu")) == "cuda" else -1
    n = min(len(gen_texts), len(ref_texts))
    if n < 16:
        return {"mauve": None, "error": f"need >=16 paired texts, have {n}"}
    gen, ref = gen_texts[:n], ref_texts[:n]
    for fm in featurizers:
        try:
            out = mauve.compute_mauve(p_text=gen, q_text=ref, device_id=device_id,
                                      featurize_model_name=fm, max_text_length=max_text_length,
                                      verbose=False)
            return {"mauve": float(out.mauve), "featurizer": fm}
        except Exception as e:
            last = str(e)
    return {"mauve": None, "error": f"all featurizers failed: {last}"}
