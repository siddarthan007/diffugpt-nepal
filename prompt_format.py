"""Shared chat prompt format for instruction-tuning + serving.

The model is a masked-diffusion LM, so "chat" = give the prompt as CLEAN anchor tokens
and let the model denoise the answer region (LLaDA-style: during SFT only the response
is masked/supervised; the prompt is never masked).

IMPORTANT: the tokenizer is Devanagari-only with byte_fallback. ASCII characters like
'#', ':' and '\\n' have NO real tokens and fall back to raw byte tokens (<0x23> etc.),
which both display badly and give the model out-of-distribution anchors. So the template
uses ONLY Devanagari words + spaces: प्रश्नः ("question:") and उत्तरः ("answer:").
"""

PROMPT_TEMPLATE = "प्रश्नः {instruction} उत्तरः "
RESPONSE_MARKER = "उत्तरः"


def build_prompt_ids(sp, instruction):
    """Token ids for the prompt prefix (everything up to and including the answer marker)."""
    return sp.encode(PROMPT_TEMPLATE.format(instruction=instruction.strip()))


def build_example_ids(sp, instruction, answer, eod_id):
    """(full_ids, response_mask) for one SFT example. response_mask[i]=True where the
    token is part of the answer (the only positions masked + supervised)."""
    prompt_ids = build_prompt_ids(sp, instruction)
    answer_ids = sp.encode(answer.strip()) + [eod_id]
    full = prompt_ids + answer_ids
    mask = [False] * len(prompt_ids) + [True] * len(answer_ids)
    return full, mask


def strip_response(text):
    """For display fallback: keep only what follows the answer marker. (Serving prefers
    position-based extraction via the anchor mask, which is collision-proof.)"""
    i = text.rfind(RESPONSE_MARKER)
    if i >= 0:
        text = text[i + len(RESPONSE_MARKER):]
    return text.strip()
