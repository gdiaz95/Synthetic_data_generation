import importlib
import os


class _NullRun:
    def __init__(self):
        self.id = "disabled"
        self.resumed = False
        self.name = "wandb-disabled"

    def log(self, *_args, **_kwargs):
        return None

    def finish(self):
        return None


class _NullWandb:
    def init(self, *_args, **_kwargs):
        return _NullRun()

    def log(self, *_args, **_kwargs):
        return None

    def Table(self, dataframe=None, **_kwargs):
        return dataframe

    def Image(self, image=None, **_kwargs):
        return image


def _read_enabled_default():
    value = os.getenv("SYNTHGEN_ENABLE_WANDB", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_wandb_client(enabled=None):
    if enabled is None:
        enabled = _read_enabled_default()

    if not enabled:
        print("W&B logging is disabled. Results are still saved to JSON reports.")
        return _NullWandb()

    try:
        return importlib.import_module("wandb")
    except ModuleNotFoundError:
        print("W&B requested but package is not installed. Continuing without W&B logging.")
        return _NullWandb()
