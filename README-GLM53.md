# EXO for GLM-5.3-Flash

A fork of [exo-explore/exo](https://github.com/exo-explore/exo) **v1.0.71** that
runs two GLM-5.3-Flash builds out of the box, pipeline-sharded across a cluster of
Apple-silicon Macs connected over Thunderbolt RDMA. Upstream exo cannot load this
architecture at all (`Model type glm5_next not supported`), and the release that
could load it produced silent garbage. Everything below is what it took to make
it work, packaged so nothing has to be patched by hand.

## The two models

| Model | Quant | On disk | Layers | Sharding | Cluster it was validated on |
| --- | --- | --- | --- | --- | --- |
| [`Vontra/GLM-5.3-Flash-MLX-oQ2-MTP`](https://huggingface.co/Vontra/GLM-5.3-Flash-MLX-oQ2-MTP) | 2-bit (8-bit attention) | 110 GB | 45 | Pipeline | 4x Mac Studio, 96 GB each |
| [`orcarouter/GLM-5.3-Flash-Uncensored-MLX-6bit`](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-MLX) | 6-bit (abliterated) | 289 GB | 45 | Pipeline | 4x Mac Studio, 96 GB each |

Both are `glm5_next`: a hybrid of Gated-DeltaNet linear attention, DeepSeek-style
sparse attention and MoE. Neither supports tensor parallelism, so they shard by
layers; the 6-bit build needs roughly 82 GB on its heaviest rank and therefore
four 96 GB machines. Both ship here as built-in model cards, so they appear in the
dashboard and can be placed without registering anything.

Measured on the validation cluster (4-node pipeline over RDMA): the 2-bit decodes
at 22-25 tok/s with ~0.6 s time-to-first-token; the 6-bit at ~15-20 tok/s single
stream, ~31 tok/s aggregate over four concurrent requests, and reads a 200k-token
prompt in about 3.9 minutes.

## Install

Download `EXO-1.0.71-glm53.dmg` from the
[Releases](https://github.com/zhast/exo/releases) page and drag `EXO.app` to
Applications, on every machine in the cluster.

The build is **not signed or notarized** (that needs an Apple Developer identity
this fork does not have), so Gatekeeper will refuse to open it the first time.
Either right-click `EXO.app` and choose *Open*, or clear the quarantine flag:

```bash
xattr -d com.apple.quarantine /Applications/EXO.app
```

To build it yourself instead, from a checkout of the `glm53-release` branch:

```bash
just build-app
```

That compiles the Rust networking crate, installs the pinned dependencies
(including the patched `mlx-lm` and `mlx-vlm` forks described below), packages
the Python runtime with PyInstaller and builds `EXO.app` with Xcode.

## Run

Start EXO on each node. They discover each other over mDNS on the local network
and, when every pair of machines is linked by Thunderbolt, form a full RDMA mesh.
Then place a model from the dashboard, or from the API:

```bash
curl -s http://<any-node>:52415/instance/previews | jq .
# take the Pipeline/MlxJaccl preview for the model you want and POST it back
curl -s -X POST http://<any-node>:52415/instance -H 'Content-Type: application/json' -d @preview.json
```

`MlxJaccl` (RDMA over Thunderbolt) needs the *complete* mesh: every pair of nodes
must have a trained Thunderbolt link. If any link is missing, placement falls back
to `MlxRing` (TCP over your LAN), which works but is roughly three times slower.
The Thunderbolt probe is given 240 s here because on a Mac Studio with many
ports it takes well over the 30 s upstream allowed, and a timed-out probe used to
make every RDMA placement fail silently.

Both models' chat templates default to maximum reasoning effort and only
recognise `low` and `high`, so send `reasoning_effort` per request; `low` answers
short tasks in about a second.

## What is patched

### In exo (this branch, on top of v1.0.71)

| Change | Upstream |
| --- | --- |
| Re-derive layer indices after a pipeline split, so a rank's sparse- and linear-attention layer lists match the layers it actually holds | [exo#2287](https://github.com/exo-explore/exo/pull/2287) |
| Reject image input when the instance has no vision processor, instead of answering about an image it never saw | [exo#2293](https://github.com/exo-explore/exo/pull/2293) |
| Bound SIGKILL retries so a runner wedged in an uninterruptible kernel wait cannot hang the worker and the master's API | [exo#2294](https://github.com/exo-explore/exo/pull/2294) |
| Give the Thunderbolt `system_profiler` probe 240 s so RDMA edges are actually published | from [exo#2286](https://github.com/exo-explore/exo/pull/2286) |
| Start a leader election only when the connected-peer set changes (libp2p's 5 s re-dial of every peer otherwise paused the API ~3 s, twelve times a minute) | local |
| Keep SSM snapshots on a stride plus the two most recent, and prune off-grid snapshots on prefix-cache updates, so a 200k-token prompt no longer pins ~5 GB per rank | local |
| Cap MLX's free-buffer cache, clear it when the batch drains, refuse prompts over a per-deployment token limit, and expose these plus the prefill chunk size as tunables | local |
| Return a proper error body on the non-streaming path instead of truncating the response | local |
| Built-in model cards for both GLM-5.3-Flash builds | local |

The Rust multicast-discovery fix that is also part of exo#2286 targets the
zenoh-based networking that replaced libp2p *after* v1.0.71; it does not apply to
this release, which discovers over libp2p mDNS.

### In the dependencies (pinned forks)

exo loads every text model through `mlx-lm`, which has no `glm5_next`, and runs
the model through `mlx-vlm`. This branch pins both to forks that carry the fixes
as commits, so `uv sync` and the packaged app include them:

| Package | Pinned to | Carries |
| --- | --- | --- |
| `mlx-lm` | [`zhast/mlx-lm` `glm53-release`](https://github.com/zhast/mlx-lm/tree/glm53-release) (`4dbede85`, on `rltakashige/mlx-lm` `leo/deepseek-v4`) | A `glm5_next` bridge that adapts mlx-vlm's language model to mlx-lm's loader, re-nests the forget-gate quantization tensors, canonicalises upstream-HF-layout checkpoints, and casts float16 scales to bfloat16 (float16 scales otherwise promote activations to float32 and any prompt of nine tokens or more dies in a Metal kernel that exceeds threadgroup memory); a nesting-aware quantization predicate |
| `mlx-vlm` | [`zhast/mlx-vlm` `glm53-release`](https://github.com/zhast/mlx-vlm/tree/glm53-release) (`95548557`, on v0.6.17) | The hyper-connection float32 cast ([mlx-vlm#2139](https://github.com/Blaizzy/mlx-vlm/pull/2139) — the kernel type-punned bfloat16 as float32 and the model emitted plausible garbage at full speed); absorbed-MLA prefill and a true sparse prefill (O(T) instead of O(T²), 3.7x faster at 200k tokens); a DSA-indexer staleness guard. `mlx-audio` is made optional (its transformers floor is stricter than the code needs). |

`transformers` is pinned to 5.2.0, the release the validation cluster runs.
Reference diffs for all of these are in [`patches/`](patches/).

## Tunables

Set as environment variables, or in an optional
`/usr/local/etc/exo-tunables.json` that takes effect on the next instance reload
(environment variables win):

| Tunable | Env | Default | Notes |
| --- | --- | --- | --- |
| `prefill_step_size` | `EXO_PREFILL_STEP_SIZE` | 1024 | 4096 upstream; smaller cuts transient GPU memory on heavy ranks |
| `max_prompt_tokens` | `EXO_MAX_PROMPT_TOKENS` | 65536 | The 6-bit build is validated to 200000 on 4x96 GB with the cache limit below |
| `max_prompt_tokens_ring` | `EXO_MAX_PROMPT_TOKENS_RING` | 65536 | Applied on top of the above only when the instance runs on `MlxRing`, whose ranks hold more resident memory: a ~99k-token prompt killed the heaviest 6-bit rank at ~90 GB on the TCP fallback while 200k is fine over RDMA |
| `mlx_cache_limit_gb` | `EXO_MLX_CACHE_LIMIT_GB` | unset | 2.0 on the validation cluster; cached-but-free Metal buffers stay wired |
| `ssm_snapshot_stride_tokens` | `EXO_SSM_SNAPSHOT_STRIDE` | 8192 | Coarser costs at most this many tokens of re-prefill on a partial prefix hit |
| `mem_telemetry` | `EXO_MEM_TELEMETRY` | off | Logs MLX active/peak/cache per request |

## Caveats

- This is a Debug-configuration, unsigned build made with the repository's own
  `just build-app` target. It is functionally the same runtime the validation
  cluster serves from; it is not the signed, auto-updating app upstream ships.
- Never register the 6-bit build under the real repository id
  (`orcarouter/GLM-5.3-Flash-Uncensored-MLX`): that repository's root
  `config.json` is the 4-bit variant and would overwrite the 6-bit one. The
  built-in card uses the synthetic `-6bit` id for that reason, and the id must
  keep the literal `glm-5` substring or the wrong stop tokens are chosen.
- A `system_profiler` Thunderbolt link that fails to enumerate blocks RDMA
  placement even when the interface is up; re-seat the cable or reboot that node.
- The upstream pull requests above are all still open. This fork carries them
  until they merge.
