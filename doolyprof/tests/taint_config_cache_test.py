"""A tainted config must return what was last written, not a cached tainted copy.

vLLM 0.32+ reads ``sliding_window`` and, when a checkpoint stores 0 (transformers
5.x does for Qwen1.5-MoE), writes ``sliding_window = None``. The tracer cached the
first read, so vLLM kept seeing 0, derived max_model_len = 0 and refused to start.

Runs directly: ``python doolyprof/tests/taint_config_cache_test.py``. CPU only.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from transformers import MistralConfig  # noqa: E402

from doolyprof.tracer.config import create_tainted_config, _TaintedDict  # noqa: E402
from doolyprof.tracer.types import TaintedInt  # noqa: E402


def test_attribute_write_after_read():
    # sliding_window is optional (int | None) in every transformers version, as
    # in the vLLM case this reproduces.
    cfg = create_tainted_config(MistralConfig(hidden_size=64, intermediate_size=128, num_attention_heads=4,
                                              num_key_value_heads=4, num_hidden_layers=2, sliding_window=4096))
    first = cfg.sliding_window
    assert isinstance(first, TaintedInt) and int(first) == 4096, repr(first)
    cfg.sliding_window = None
    assert cfg.sliding_window is None, repr(cfg.sliding_window)
    cfg.sliding_window = 2048
    assert int(cfg.sliding_window) == 2048, repr(cfg.sliding_window)
    assert isinstance(cfg.sliding_window, TaintedInt)


def test_dict_write_after_read():
    d = _TaintedDict({"factor": 8, "original_max_position_embeddings": 8192})
    assert int(d["factor"]) == 8
    d["factor"] = 4
    assert int(d["factor"]) == 4, d["factor"]
    d.update({"factor": 2})
    assert int(d["factor"]) == 2, d["factor"]
    assert int(d.pop("factor")) == 2
    assert "factor" not in d


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
