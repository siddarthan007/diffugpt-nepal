"""Shared chat prompt format for instruction-tuning + serving.

The model is a masked-diffusion LM, so "chat" = give the prompt as CLEAN anchor tokens
and let the model denoise the answer region (LLaDA-style: during SFT only the response
is masked/supervised; the prompt is never masked).
"""

# निर्देशन = "instruction", उत्तर = "answer". Plain text markers (tokenized normally).
PROMPT_TEMPLATE = "### निर्देशन:\n{instruction}\n\n### उत्तर:\n"
RESPONSE_MARKER = "### उत्तर:\n"


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
    """For display: keep only what follows the answer marker."""
    i = text.rfind("### उत्तर:")
    if i >= 0:
        text = text[i + len("### उत्तर:"):]
    return text.lstrip("\n").strip()
