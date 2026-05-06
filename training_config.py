import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def project_path(*parts):
    return str(BASE_DIR.joinpath(*parts))


def base_model_path():
    return os.environ.get("GPT2_MODEL_PATH", project_path("gpt2"))


def int_env(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def training_knobs(default_batch_size=2, default_grad_accum=4):
    return {
        "per_device_train_batch_size": int_env("TRAIN_BATCH_SIZE", default_batch_size),
        "gradient_accumulation_steps": int_env("GRAD_ACCUM_STEPS", default_grad_accum),
        "max_steps": int_env("TRAIN_MAX_STEPS", -1),
        "gradient_checkpointing": gradient_checkpointing_enabled(),
    }


def gradient_checkpointing_enabled():
    return os.environ.get("DISABLE_GRADIENT_CHECKPOINTING") != "1"


def model_dir_is_loadable(model_dir, model_class):
    try:
        model_class.from_pretrained(str(model_dir))
    except Exception:
        return False
    return True


def apply_model_memory_settings(model):
    model.config.use_cache = False
    if gradient_checkpointing_enabled():
        model.gradient_checkpointing_enable()
    return model
