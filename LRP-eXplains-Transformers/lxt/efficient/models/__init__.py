import lxt.efficient.models.gpt2 as gpt2

# Only GPT-2 is loaded here; other models (qwen3, gemma3, etc.) require
# transformers >=5.x and are skipped for this project.
try:
    import lxt.efficient.models.llama as llama
    _llama = {llama.modeling_llama: llama.attnLRP}
except Exception:
    _llama = {}

try:
    import lxt.efficient.models.bert as bert
    _bert = {bert.modeling_bert: bert.attnLRP}
except Exception:
    _bert = {}

DEFAULT_MAP = {
    gpt2.modeling_gpt2: gpt2.attnLRP,
    **_llama,
    **_bert,
}

def get_default_map(module):
    if module in DEFAULT_MAP:
        return DEFAULT_MAP[module]
    else:
        supported_models = ", ".join([key.__name__ for key in DEFAULT_MAP.keys()])
        raise ValueError(f"{module.__name__} not yet supported. Supported models are: {supported_models} " \
                         f"Please provide a custom 'patch_map'. Contributions to the GitHub repository are welcome!")
                 


