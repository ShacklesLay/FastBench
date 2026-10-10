# MOSS-VL-Realtime: native framewise FastBench adapter

This opt-in adapter is based on official FastBench commit
`277ab8826d943e62c8b7a9fefc44b229bfe55d2b`. Existing evaluator, model server,
configs, scripts and other model paths are unchanged. The original content judge
and answer-selection/timing branches are reused. This is a proposed native
adaptation plus an explicitly labelled fractional input-event clock, not a
confirmed reproduction of the paper's Table 1 conditions.

The system prompt corresponds to the [public MOSS realtime example](https://github.com/OpenMOSS/MOSS-VL/blob/f4ffcd3658fa9cc3a288d05ff9a705fbc7778c28/inference/realtime/run_online_inference.py#L39-L44);
`prompt_source.json` records the exact text and the Markdown-backtick formatting
difference. This references prompt text only, without switching to a newer runtime.

## Fixed protocol

All three task types use the same rules:

- Frozen realtime system prompt (`realtime_system_prompt.txt`, SHA256 `3e438e9c992bd290d70801ae43ebb8144642394d300ce6a8de626503b01d94d9`).
- Original native 2fps sampler and decoder pixels; watch from video start through
  the real end, including any final partial chunk. No GT-triggered active windows.
- Each visual input has exactly one image. After model consumption, advance only
  on a **fresh generated** `<|silence|>` (151671). Text before silence is retained.
  Historical frames follow the same rule; no prefilling synthetic silence.
- At the original integer question time, inject a pure-text question (zero images)
  before the first native chunk at that time. Release the question on the first
  actual native-decoded non-control text chunk or generated silence; then submit
  the next native frame. No extra Q frame, duplication removal or early question.
- Visual watchdog 60s; question watchdog 30s. Timeout is execution failure and
  blocks scoring, not a normal zero or permission to advance. Token counts do not release the gate.
- Keep every native FIFO chunk. Reconstruct origin slots, including late question
  chunks collected in a subsequent HTTP. Keep original HTTP raw separately.
- A visual slot is anchored to its actual model-visible native timestamp text
  (`<|time_start|>11.5 seconds<|time_end|>`); a question slot to original arrival.
  Preserve fractional seconds. This clock is **not wall-clock latency**.
- Upstream strict scoring is the default: same first-eligible-answer, placeholders,
  early/late rules, reference answers and content judge. Only the timestamp parser
  accepts fractions. In particular Q==A still has the original strict cutoff.
  No best-answer selection, GT-conditioned timestamp or hidden four-second relaxation.

The default adapter includes final partial chunks and all questions in a video.
The saved split runs below have different task scopes and tail coverage; these
outputs do not replace a new joint evaluation of the default adapter.

## Evidence and validation status

`evidence/` includes sanitized 306-QA details and the measured split-run summary:
Forward 44.047619 / Instant 46.849315 / Backward 41.846154,
weighted mean 44.248366. This is **not a new joint full-run score**. Forward stopped
at the final complete second; I/B included the real partial tail. Separate task
runs also exclude the other types' questions in three shared videos. The default
adapter below evaluates all questions together and needs its own author rerun.

We replayed all 32 saved Forward and Instant/Backward worker traces (27,778 input slots) through the
portable FIFO/scoring implementation with exact stored judgments: all 306 selected
texts, anchors, categories, scores and explanations match. Twelve new CPU regression
tests plus fourteen existing upstream tests pass. A real-tokenizer/calendar/pixel
check on nine existing I/B videos passes 819 single visual+9 pure question slots.
No new model/judge calls were made for these validations. The refactored portable
server/client have **not** received a fresh standalone GPU smoke/full run.
Start with smoke, then perform the author's independently scored full rerun.

Our judge was local BF16 vLLM Qwen3-235B-A22B-Instruct-2507 rather than author
OpenRouter. Inference used torch2.8/torchvision0.23/torchcodec0.7 rather than the
upstream2.7.1/0.22.1 requirements. `tested_environment.json` records exact versions.
Content judgments used in the Forward time reconstruction were reused only for
exact identical inputs, by a predetermined source priority, never by highest score.

## Reuse our actual intermediate outputs

`evidence/runs/forward/` contains the actual 167-video/168-QA Forward run;
`evidence/runs/instant_backward/` contains the actual 136-video/138-QA I/B run.
Each includes every original HTTP response (`raw_http.jsonl`), lossless generating
slots/chunks (`slots.jsonl`), fractional-slot predictions (`predictions.jsonl`),
selected/scored rows (`result_details.jsonl`), and export receipts with source hashes.
Private media locations are replaced with the public dataset-relative `video_path`;
the unused private `stream_addr` is removed. Questions, GT, outputs and timing
records are unchanged. Videos and weights are not distributed in this fork.

To verify the saved results locally, without model inference or judge API calls:

```bash
python -m moss_native.replay_evidence
```

If inference resources are unavailable, authors can independently rejudge the
actual predictions with their configured original judge:

```bash
python -m moss_native.score --run moss_native/evidence/runs/forward \
  --output runs/rejudge-forward --num-scorers 16
python -m moss_native.score --run moss_native/evidence/runs/instant_backward \
  --output runs/rejudge-ib --num-scorers 16
```

Configure the original Qwen judge as below first. These are subset runs: omit
`--require-full`, retain both task scopes, and label their weighted aggregate as
split-run evidence. Three shared videos have different subset question contexts;
do not merge these exports into a fictitious joint 300-video run. Reusing saved
judgments reproduces our reported result; independent rejudging may differ.

## Setup

Run commands at the FastBench repository root. Use the pinned public snapshots:

```bash
hf download OpenMOSS-Team/MOSS-VL-Realtime \
  --revision d1f71a588deaf2b9b9cea7c6fff66cb10847cc86 --local-dir checkpoints/moss-vl-realtime
hf download Ashenone3/FastBench --repo-type dataset \
  --revision 818f572cc9f402bc1b4b3d83ec7e084483730131 --local-dir dataset
python -m moss_native.check_model --model-path checkpoints/moss-vl-realtime --weights
```

Use Python3.12, CUDA and ffmpeg/ffprobe. To reproduce our recorded native runtime,
install `requirements-moss-native.txt` in an isolated environment; select the
CUDA12.8 wheel index for torch/torchvision:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-moss-native.txt
```

This optional requirements file does not replace the upstream environment.
Authors should record any environment differences in their rerun. Checkpoint
code/config/tokenizer hashes are checked by the server; use the separate
`--weights` check once before deploying replicas to validate all weight shards.

## One-GPU smoke, then full inference

Use a fresh trace and output directory for every attempt. The server and client
need access to the same frame files and trace directory. One server owns one
sequential inference shard. Never concurrently share that stateful server across
shards. The default attention implementation is sdpa, as in our recorded run.

Terminal 1:

```bash
CUDA_VISIBLE_DEVICES=0 MODEL_PATH="$PWD/checkpoints/moss-vl-realtime" \
  TRACE_DIR="$PWD/runs/native-smoke-trace" PORT=8000 \
  bash scripts/deploy_moss_vl_framewise.sh
```

Terminal 2, after readiness:

```bash
MODEL_PATH="$PWD/checkpoints/moss-vl-realtime" TRACE_DIR="$PWD/runs/native-smoke-trace" \
  OUTPUT_DIR="$PWD/runs/native-smoke" VIDEO_ROOT="$PWD" \
  bash scripts/eval_moss_vl_framewise.sh --max-samples 3
```

Smoke is unscored. Inspect `receipt.json` and `audit.json`, then stop the server
normally. Repeat with fresh paths and remove `--max-samples` for all 300 videos/306 QA.
`--scope forward` or `--scope instant_backward` provides explicitly labelled
subset diagnostics; default is **all** questions together. The scheduling state
strips gold response/time/type/focus; the separate raw retains gold for scoring.

## Independent single-GPU shards

Deploy one server per independently allocated GPU, each with unique port/trace
path. Each client receives `--num-shards N --shard-index I` (I=0..N-1) over the
same dataset and **all** scope. Sequentially handle that shard's videos; no server
is shared with another shard. This repository allocates no GPUs automatically.
After all shards are validated:

```bash
python -m moss_native.merge --runs runs/native-w000 runs/native-w001 \
  --output runs/native-merged
```

The merger requires all declared disjoint shards,300 IDs/306 QA,matching protocol,
dataset/config hashes and no API failures. It rejects smoke or mixed task scopes.
For a single complete run no merge is needed. Existing outputs are never overwritten.

## Score with the author's judge

Reuse the upstream judge configuration/environment (no keys in scripts or artifacts):

```bash
export STREAMEVAL_JUDGER_BACKEND=openrouter
export STREAMEVAL_JUDGER_API_BASE=https://openrouter.ai/api/v1
export STREAMEVAL_JUDGER_MODEL=qwen/qwen3-235b-a22b-2507
# Set STREAMEVAL_JUDGER_API_KEY in your own shell using the upstream instructions.
python -m moss_native.score --run runs/native-merged \
  --output runs/native-score --num-scorers 16 --require-full
```

A local OpenAI-compatible backend is supported by the original configuration;
record backend/model differences. The default YAML fallback judge is deliberately
rejected to avoid accidentally using the model inference endpoint as the judge.

The scorer uses original LLMJudger/objective branches and judges each identical
(question, selected response, reference) once. Every actual request/reply is saved
without API headers/keys; failures abort score acceptance. `--require-full`
requires 168 Forward / 73 Instant / 65 Backward across300 videos. Without it, the mean is
explicitly a subset mean. Early/late/NoResponse categories and exact response
anchors are retained in `result_details.jsonl`.

## Tests and upstream review

```bash
python -m unittest discover -s tests -v
```

Please assess whether this native adaptation and its disclosed input-event clock
fit FastBench's intended evaluation design. If so, please use the supplied scripts
for a rerun and update the relevant paper/repository results. If inference resources
are limited, the complete actual intermediate predictions are available for
independent scoring/review, with their split-run scope clearly labelled.
