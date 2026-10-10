# Public MOSS/JoyAI paths and our native adaptation

This describes the frozen public code, not an assertion about undisclosed Table 1
commands. Base: `277ab8826d943e62c8b7a9fefc44b229bfe55d2b`.

| Aspect | Public MOSS path | Public JoyAI path | Supplied MOSS adapter |
|---|---|---|---|
| History organization | Accumulates pending frames/prompts outside active windows, then publishes them to a persistent realtime session | Records each second as a chronological user turn and assistant output; skipped generation explicitly records `</silence>` | All original frames arrive individually from video start; native generated text/silence remains in context |
| Visual progression | Enqueues all frames in one request without waiting for a fresh generated silence after each | One-second image groups; synthetic silence in skipped turns | One image at a time, wait for fresh generated silence after consumption |
| Question organization | Enqueues frames then prompts; the native processor may drain them together | Query text precedes the current chunk images in the user turn | Pure text question at original arrival, before the next original causal frame |
| Output clock | Integer right-edge HTTP clock | Same integer HTTP clock | Separate fractional input-event slot clock; original HTTP raw also preserved |

Our usual MOSS evaluation resembles the **chronological interaction organization**
of native interaction models. JoyAI's current public path is not our identical
single-frame generated-silence protocol; it also has its own context compaction
and native token format. The common integer clock is not a MOSS-only issue.

Source: [`_run_sample_native_moss`](https://github.com/Ashone3/FastBench/blob/277ab8826d943e62c8b7a9fefc44b229bfe55d2b/run_stream_eval.py#L3375-L3469),
[`_run_sample_native_joyai`](https://github.com/Ashone3/FastBench/blob/277ab8826d943e62c8b7a9fefc44b229bfe55d2b/run_stream_eval.py#L3586-L3737),
and [`MossVLEngine._generate_realtime`](https://github.com/Ashone3/FastBench/blob/277ab8826d943e62c8b7a9fefc44b229bfe55d2b/hf_openai_server.py#L638-L704).

Initial MOSS reproduction: overall29.738562 / Forward34.523810 /
Instant29.863014 / Backward17.230769, compared with paper27.9 /36.1 /22.2 /13.2.
Native split runs: overall44.248366 / Forward44.047619 / Instant46.849315 /
Backward41.846154. The cumulative delta includes prompt, input calendar/layout and
clock changes; it is not an isolated model-ability gain. All responses/selection
evidence are in `evidence/runs/`, and scope/environment differences are in the README.
