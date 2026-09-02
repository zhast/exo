# Out-of-tree patches: running GLM-5.3-Flash (glm5_next) on a Thunderbolt RDMA cluster

These are changes to third-party packages inside the venv, not to exo. They are
what it took to load and run `Vontra/GLM-5.3-Flash-MLX-oQ2-MTP` (102.6 GiB,
`glm5_next`: hybrid Gated-DeltaNet linear attention + DeepSeek sparse attention
+ MoE, 45 layers) pipeline-sharded over `MlxJaccl` on 4x Mac Studio.

Verified against EXO **v1.0.71 built from source** (not the shipped .app, whose
Python is sealed in a PyInstaller archive and cannot accept a new architecture).
On that build the cluster federates over libp2p with `--bootstrap-peers` and
forms a full RDMA mesh; `Pipeline/MlxJaccl` reaches `ready=3/3` and generates.

| file | what it fixes |
| --- | --- |
| `mlx_lm-models-glm5_next.py` | NEW. Bridges `glm5_next` (which exists only in mlx-vlm) into mlx-lm's registry, because EXO loads text models through mlx-lm unconditionally. Also (a) re-nests the forget-gate `f_a_proj`/`f_b_proj` `.scales`/`.biases`, which mlx-vlm's `sanitize` leaves un-nested on a quantized checkpoint, (b) builds caches from `mlx_lm.models.cache` classes, and (c) canonicalises checkpoints quantised in the upstream-HF layout (`model.language_model.layers.N.*`, per-expert `mlp.experts.N.*`, raw `kv_b_proj` — e.g. `orcarouter/GLM-5.3-Flash-Uncensored-MLX`) to the `model.layers.N.*` prefix mlx-vlm's inherited DeepSeek-V32 `sanitize` keys its expert stacking and `kv_b_proj` absorption on. |
| `mlx_vlm-glm5_next-language.patch` | `ssm_idx`/`fa_idx` as properties resolved against the *current* layer list, plus a `_pool` staleness guard for the DSA indexer. |
| `mlx_lm-utils-nested-quant.patch` | Makes `load_model`'s `class_predicate` nesting-aware so `forget_gate.f_a_proj` finds its un-nested `config["quantization"]` override, and lets a module at `language_model.model.layers.N.x` find an override written as `model.layers.N.x` (how upstream-HF-layout checkpoints key their per-module bit widths). |

## Why the cache-class fix is needed

mlx-lm and mlx-vlm ship API-identical duplicates of `ArraysCache` / `CacheList` /
`KVCache`. EXO detects recurrent (SSM) layers with `isinstance` against
**mlx-lm's** `ArraysCache` — in `has_non_kv_caches()`, the post-prefill rollback
(`generate.py:381`) and `cache.py`'s trim path. Caches built by mlx-vlm's
`make_cache` fail every one of those checks, so EXO treats recurrent state as a
KV cache and calls `.trim()` on it:

    AttributeError: 'ArraysCache' object has no attribute 'trim'

This blocked `RunnerReady` entirely.

## Why the index fix is needed

`Glm5NextModel` cached `ssm_idx`/`fa_idx` at `__init__` over all 45 layers. EXO's
pipeline sharding replaces `self.layers` with a slice afterwards, so those
indices address the wrong entry of the per-shard cache — `create_ssm_mask()`
receives a full-attention cache and vice versa. On the real 15/15/15 split, two
of three shards had **both** masks inverted. The old `next(..., 0)` default also
silently returned 0 for a shard containing no layer of that type.

Note: this is latent for single-turn requests (at prefill every cache has
offset 0, so the wrong index yields an identical mask). It bites on N>1 tokens
at KV offset>0 — multi-turn, or prefix-cache reuse.

## Status

Loading, sharding, RDMA transport and generation all work. Output quality is
still wrong on this checkpoint: it emits valid vocabulary with no semantics.
A control on the **same** 3-node MlxJaccl pipeline is coherent
(`mlx-community/GLM-4.7-Flash-4bit` answers `391` and `Paris`), so the pipeline
and RDMA path are not implicated. Unresolved: whether the `oQ2` 2-bit community
quant is itself broken, or the bridge has a numerical bug. mlx-vlm's own loader
cannot load this checkpoint at all (it rejects 483 parameters), so there is no
native reference to diff against. Testing a 4-bit build of the same model is the
experiment that separates the two.
