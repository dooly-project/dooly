"""Unit tests for MoE forced routing and its sweep grids (profiler/moe_hook.py).

* Without expert parallelism the forced ids are the 2-D (num_tokens,
  activated_experts) cycle used before, unchanged.
* With expert parallelism the 3-D grid adds ``local_pairs``: the forced routing
  puts exactly that many (token, expert) pairs on ``activated_experts`` local
  experts, keeps every token's picks distinct, and sends all other pairs to
  experts the layer's ``expert_map`` marks remote (-1).

Runs directly: ``python doolyprof/tests/moe_routing_test.py``. CPU only.
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch  # noqa: E402

from doolyprof.profiler import moe_hook  # noqa: E402


class _Router:
    def _compute_routing(self, hidden_states, router_logits, indices_type=None, **kw):
        raise AssertionError("not forced")


class _MoE(torch.nn.Module):
    """Only what force_moe_routing reads from a vLLM MoE layer."""

    def __init__(self, num_experts, top_k, ep_size=1):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        local = num_experts // ep_size
        self.moe_config = types.SimpleNamespace(experts_per_token=top_k, num_local_experts=local)
        self.router = _Router()
        if ep_size > 1:  # rank 0 under linear placement, as determine_expert_map builds it
            m = torch.full((num_experts,), -1, dtype=torch.int32)
            m[:local] = torch.arange(local, dtype=torch.int32)
            self.expert_map = m
        else:
            self.expert_map = None


def _routed_ids(layer, num_tokens, activated, local_pairs=None):
    with moe_hook.force_moe_routing(layer, num_tokens, activated, local_pairs=local_pairs):
        _, ids = layer.router._compute_routing(None, None, torch.int64)
    return ids.tolist()


def test_non_ep_routing_unchanged():
    layer = _MoE(num_experts=64, top_k=6)
    ids = _routed_ids(layer, num_tokens=5, activated=16)
    assert ids == moe_hook._cycle_ids(5, 6, 16), ids
    assert moe_hook.moe_expert_placement(layer) == (list(range(64)), [])


def test_ep_local_pairs_routing():
    layer = _MoE(num_experts=64, top_k=6, ep_size=2)
    local, remote = moe_hook.moe_expert_placement(layer)
    assert local == list(range(32)) and remote == list(range(32, 64))
    for T, L, A in [(5, 7, 4), (5, 30, 8), (5, 1, 1), (3, 18, 16), (8, 13, 2), (5, 30, 30)]:
        ids = _routed_ids(layer, T, A, local_pairs=L)
        flat = [e for row in ids for e in row]
        local_hits = [e for e in flat if layer.expert_map[e] >= 0]
        assert len(local_hits) == L, (T, L, A, ids)
        assert len(set(local_hits)) == min(A, L), (T, L, A, ids)
        assert all(len(set(row)) == 6 for row in ids), ids
        assert len(flat) - L == sum(1 for e in flat if layer.expert_map[e] < 0)


def test_ep_grid_bounds():
    grid = moe_hook.moe_ep_grid(top_k=6, num_local_experts=32, num_tokens=5)
    assert (30, 32) not in grid and max(p for p, _ in grid) == 30
    for pairs, act in grid:
        assert 1 <= pairs <= 30
        assert -(-pairs // 5) <= act <= min(32, pairs), (pairs, act)
    assert (1, 1) in grid and (30, 30) in grid and (30, 8) in grid and (30, 4) not in grid


def test_ep_routing_rejects_impossible_configs():
    layer = _MoE(num_experts=64, top_k=6, ep_size=2)
    for T, L, A in [(5, 31, 8), (5, 10, 1), (5, 4, 5)]:
        try:
            _routed_ids(layer, T, A, local_pairs=L)
        except ValueError:
            continue
        raise AssertionError(f"accepted T={T} L={L} A={A}")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                failures += 1; print(f"FAIL {name}: {e}")
            except Exception as e:  # noqa: BLE001
                failures += 1; print(f"ERROR {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
