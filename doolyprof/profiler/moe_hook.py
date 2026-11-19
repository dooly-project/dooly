"""Model-agnostic MoE forced routing +
the activated_experts grid, for the 2-D (num_tokens, activated_experts) MoE profile.

Every vLLM MoE model routes its expert computation through one common layer that
owns the router and an engine-defined ``FusedMoEConfig`` (``moe_config``): the
``FusedMoE`` class (or its ``SharedFusedMoE`` subclass) up to vLLM 0.17, the
``MoERunner`` built by ``FusedMoEFactory`` in vLLM 0.32. Finding that layer by
its ``moe_config`` covers Mixtral, Qwen2/3-MoE, Phi-MoE, DeepSeek-V2, GPT-OSS,
Llama-4, ... with no per-model or per-release class name. We force the router
*instance's* `_compute_routing` (reached from every router subclass via
`select_experts`) so exactly `activated_experts` distinct experts fire,
regardless of the (dummy-weight) gate -- LLMServingSim's methodology
(profiler/core/hooks/moe_hook.py), but keyed on the layer's contract rather than
a per-model class name.

Expert counts come from ``moe_config``: ``experts_per_token`` (top-k) and
``num_local_experts``, the experts this rank holds. Which global expert ids are
local is read from the layer's ``expert_map`` (global -> local index, -1 for
experts held by other ranks; None without expert parallelism).

Without expert parallelism every (token, expert) pair is computed here and the
grid is 2-D: (num_tokens, activated_experts). With it (``enable_expert_parallel``)
a rank computes only the pairs routed to its own experts -- the *local pairs* --
and vLLM masks the rest (expert_map -> -1). How many pairs land on a rank
depends on the routing, not on the token count alone, so the EP grid adds
``local_pairs`` as a third axis: (num_tokens, local_pairs, activated_experts),
with ``activated_experts`` counting local experts. The forced routing sends
exactly ``local_pairs`` pairs to ``activated_experts`` local experts and every
remaining pair to remote experts, which the engine masks as it would in a real
deployment.
"""
from contextlib import contextmanager

import torch


def _is_moe_layer(m) -> bool:
    try:
        from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
    except Exception:
        return False
    return isinstance(getattr(m, "moe_config", None), FusedMoEConfig) and hasattr(m, "router")


def find_fused_moe(module):
    """Return the MoE layer for ``module`` -- whether the module IS the MoE layer
    or a wrapper (e.g. Qwen2MoeSparseMoeBlock) that contains one.
    Returns None if there is no MoE layer (i.e. not an MoE module)."""
    if _is_moe_layer(module):
        return module
    if hasattr(module, "modules"):
        for m in module.modules():
            if _is_moe_layer(m):
                return m
    return None


def moe_top_k(fused) -> int:
    """Experts each token is routed to."""
    return fused.moe_config.experts_per_token


def moe_num_local_experts(fused) -> int:
    """Experts held by this rank (all of them unless expert parallelism is on)."""
    return fused.moe_config.num_local_experts


def moe_activated_grid(top_k, num_experts, num_tokens):
    """Power-of-two ``activated_experts`` grid in [top_k, min(num_experts,
    num_tokens*top_k)] -- LLMServingSim ExpertCategory.compose_shots. Always includes
    the top_k and the upper-bound endpoints so the interpolation table is well-bounded."""
    hi = min(num_experts, num_tokens * top_k)
    if hi < top_k:
        return []
    grid, v = set(), 1
    while v <= num_experts:
        if top_k <= v <= hi:
            grid.add(v)
        v *= 2
    grid.add(top_k)
    grid.add(hi)
    return sorted(grid)


def _pow2_grid(lo, hi):
    """Powers of two in [lo, hi] plus both endpoints (empty if lo > hi)."""
    if lo > hi:
        return []
    grid, v = {lo, hi}, 1
    while v <= hi:
        if v >= lo:
            grid.add(v)
        v *= 2
    return sorted(grid)


def moe_ep_grid(top_k, num_local_experts, num_tokens):
    """(local_pairs, activated_experts) grid for an expert-parallel rank.

    ``local_pairs`` L: power-of-two grid in [1, num_tokens*top_k] -- from a rank
    that receives a single pair up to one that receives every pair.
    ``activated_experts`` A (local experts that receive pairs): power-of-two
    grid in [ceil(L / num_tokens), min(num_local_experts, L)] -- at least one
    pair per activated expert, and few enough tokens per expert that each
    token's local picks are distinct experts.
    """
    out = []
    for pairs in _pow2_grid(1, num_tokens * top_k):
        lo = -(-pairs // num_tokens)
        for activated in _pow2_grid(lo, min(num_local_experts, pairs)):
            out.append((pairs, activated))
    return out


def moe_expert_placement(fused):
    """(local global ids, remote global ids) of the layer's experts, from its
    ``expert_map``; without expert parallelism every expert is local."""
    expert_map = getattr(fused, "expert_map", None)
    if expert_map is None:
        return list(range(moe_num_local_experts(fused))), []
    m = expert_map.detach().to("cpu")
    local = [i for i in torch.nonzero(m >= 0).flatten().tolist()]
    # order local ids by their local index so A experts are the first A slots
    local.sort(key=lambda g: int(m[g]))
    remote = torch.nonzero(m < 0).flatten().tolist()
    return local, remote


def _cycle_ids(num_tokens, top_k, activated):
    # ids[t, off] = (t*top_k + off) % activated  -> exactly `activated` distinct experts
    return [[(t * top_k + off) % activated for off in range(top_k)]
            for t in range(num_tokens)]


def _local_pair_ids(num_tokens, top_k, activated, local_pairs, local_ids, remote_ids):
    """Expert ids with exactly ``local_pairs`` pairs on ``activated`` local
    experts and every other pair on a remote expert. Local pairs are spread as
    evenly as possible over tokens and cycle over the first ``activated`` local
    experts; each token's picks are distinct experts."""
    base, extra = divmod(local_pairs, num_tokens)
    ids, c = [], 0
    for t in range(num_tokens):
        n_local = base + (1 if t < extra else 0)
        row = [local_ids[(c + j) % activated] for j in range(n_local)]
        c += n_local
        n_remote = top_k - n_local
        row += [remote_ids[(t * top_k + j) % len(remote_ids)] for j in range(n_remote)]
        ids.append(row)
    return ids


@contextmanager
def force_moe_routing(fused, num_tokens, activated_experts, local_pairs=None):
    """Patch ``fused.router._compute_routing`` so exactly ``activated_experts`` distinct
    experts fire for a batch of ``num_tokens`` tokens. Router-subclass-agnostic
    (FusedTopKRouter, GroupedTopKRouter, ...). Restores on exit.

    ``local_pairs`` (expert-parallel layers only): number of (token, expert)
    pairs routed to this rank's experts; the remaining pairs go to remote
    experts. None keeps every pair local (the non-EP behaviour)."""
    top_k = moe_top_k(fused)
    local_ids, remote_ids = moe_expert_placement(fused)
    if local_pairs is None:
        if not (top_k <= activated_experts <= num_tokens * top_k):
            raise ValueError(
                f"need top_k({top_k}) <= activated({activated_experts}) "
                f"<= num_tokens*top_k({num_tokens * top_k})")
        ids_list = [[local_ids[i] for i in row]
                    for row in _cycle_ids(num_tokens, top_k, activated_experts)]
    else:
        if not (1 <= local_pairs <= num_tokens * top_k):
            raise ValueError(f"need 1 <= local_pairs({local_pairs}) <= num_tokens*top_k({num_tokens * top_k})")
        if not (-(-local_pairs // num_tokens) <= activated_experts <= min(len(local_ids), local_pairs)):
            raise ValueError(
                f"need ceil(local_pairs/num_tokens) <= activated({activated_experts}) "
                f"<= min(local experts {len(local_ids)}, local_pairs {local_pairs})")
        remote_needed = top_k - local_pairs // num_tokens
        if remote_needed > 0 and len(remote_ids) < remote_needed:
            raise ValueError(f"need {remote_needed} distinct remote experts per token, layer has {len(remote_ids)}")
        ids_list = _local_pair_ids(num_tokens, top_k, activated_experts, local_pairs,
                                   local_ids, remote_ids)
    dev = next(fused.parameters()).device
    weights = torch.full((num_tokens, top_k), 1.0 / top_k, device=dev, dtype=torch.float32)
    ids_by_dtype = {}

    router = fused.router
    original = router._compute_routing

    def forced(hidden_states, router_logits, indices_type=None, *args, **kwargs):
        # indices_type is the dtype the caller wants the expert ids in (None
        # means the router's default, int32); extra keyword arguments such as
        # vLLM 0.32's input_ids are irrelevant to forced routing.
        dt = torch.int32 if indices_type is None else indices_type
        rows = hidden_states.shape[0]
        key = (dt, rows)
        if key not in ids_by_dtype:
            ids = torch.tensor(ids_list, device=dev, dtype=dt)
            w = weights
            if rows != num_tokens:
                # Routing on tokens gathered from every data-parallel rank (some
                # engines route after the EP dispatch): each rank's batch is
                # routed like this one, as fake DP assumes every rank runs it.
                if rows % num_tokens:
                    raise ValueError(f"forced routing for {num_tokens} tokens got {rows} rows")
                ids, w = ids.repeat(rows // num_tokens, 1), weights.repeat(rows // num_tokens, 1)
            ids_by_dtype[key] = (w, ids)
        return ids_by_dtype[key]

    router._compute_routing = forced
    try:
        yield
    finally:
        router._compute_routing = original
