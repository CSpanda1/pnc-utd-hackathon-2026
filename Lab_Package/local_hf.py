"""Offline Hugging Face narration using existing snapshots (no model downloads)."""
from functools import lru_cache
import os

DEFAULT_MODEL = 'Qwen/Qwen3-VL-2B-Instruct'


@lru_cache(maxsize=1)
def load_model(model_id, cache_dir, device, quantization):
    """Keep weights loaded across notebook calls and Streamlit reruns."""
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('Install requirements-hf.txt in the Python environment running this lab.') from exc
    if device not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('HF_DEVICE must be auto, cpu, or cuda')
    if quantization not in {'auto', 'none', '4bit'}:
        raise ValueError('HF_QUANTIZATION must be auto, none, or 4bit')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('HF_DEVICE=cuda requires CUDA-enabled PyTorch and a visible GPU.')
    quantization = ('4bit' if device == 'cuda' else 'none') if quantization == 'auto' else quantization
    if quantization == '4bit' and device != 'cuda':
        raise ValueError('This lab uses 4-bit quantization on CUDA only; use HF_QUANTIZATION=none for CPU.')
    local = dict(local_files_only=True, trust_remote_code=False)
    if cache_dir:
        local['cache_dir'] = cache_dir
    try:
        config = AutoConfig.from_pretrained(model_id, **local)
        tokenizer = AutoTokenizer.from_pretrained(model_id, **local)
        # These cached vision-language models also accept text-only inputs. Their
        # architecture is not registered with AutoModelForCausalLM.
        if config.model_type in {'qwen2_5_vl', 'qwen3_vl', 'qwen3_5'}:
            from transformers import AutoModelForImageTextToText
            model_class = AutoModelForImageTextToText
        else:
            model_class = AutoModelForCausalLM
        kwargs = dict(local, device_map={'': device}, dtype=torch.float16 if device == 'cuda' else torch.float32,
                      attn_implementation='eager')
        if quantization == '4bit':
            from transformers import BitsAndBytesConfig
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        model = model_class.from_pretrained(model_id, **kwargs).eval()
    except OSError as exc:
        raise RuntimeError(f'Cannot load local model {model_id!r}. Select a complete cached model or snapshot with '
                           'HF_MODEL. Run in WSL if that is where your cache lives; no download was attempted.') from exc
    return tokenizer, model


def generate_local(prompt):
    max_tokens = int(os.getenv('HF_MAX_NEW_TOKENS', '384'))
    if not 1 <= max_tokens <= 2048:
        raise ValueError('HF_MAX_NEW_TOKENS must be between 1 and 2048')
    tokenizer, model = load_model(
        os.getenv('HF_MODEL', DEFAULT_MODEL), os.getenv('HF_HUB_CACHE') or None,
        os.getenv('HF_DEVICE', 'auto'), os.getenv('HF_QUANTIZATION', 'auto'))
    import torch
    if not tokenizer.chat_template:
        raise ValueError('Choose an instruction/chat model with a chat template, such as the default Qwen model.')
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    context_limit = getattr(model.config, 'max_position_embeddings', None)
    if context_limit is None:
        context_limit = getattr(getattr(model.config, 'text_config', None), 'max_position_embeddings', None)
    if context_limit and inputs['input_ids'].shape[1] + max_tokens > context_limit:
        raise ValueError('Evidence exceeds the model context window; reduce evidence or HF_MAX_NEW_TOKENS.')
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
