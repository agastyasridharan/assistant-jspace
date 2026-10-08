# Assistant perspective in Qwen3.6-27B

Diagnostic pilot: 26 fully specified messages, each as an assistance request and under an exact-copy instruction. Eight primary contrast families, plus bereavement and frustration as auxiliary role contrasts. No study results are fabricated or imported from prior runs.

The [dashboard](https://agastyasridharan.github.io/assistant-jspace/) presents all factual pairs, exact readout anchors, J++ rankings and full O-lens outputs. Its browser checks for published updates every 30 seconds. The thread's monitoring automation syncs new results and publishes updates while the job is active.

## Frozen inputs

`data/manifest.json` contains the literal prompts, anchors, model/lens commit IDs, concept words and decoding settings. `scripts/prepare_assets.py` resolves the anchors to token IDs, audits single-token concept vocabulary, downloads the pinned J++ file, and records hashes before inference. The tokenization audit passed for all 52 records and 164 named anchors.

- Subject: `Qwen/Qwen3.6-27B`, BF16, native template, system `You are an AI assistant.`, non-thinking replies, greedy, 256-token cap.
- J++: `koayon/jpp-lenses`, block outputs 8, 16, 24, 32, 40, 44, 48, 56. FP32 transport and unembedding readout; raw and alphanumeric-filtered top 50, frozen concept logits/ranks, full user-token trajectories.
- O-lens: `ceselder/oracle-lens-qwen3.6-27b`, raw residual entering block 45, released prompt builder and overwrite injector. Ten phrases of eight tokens; greedy primary readouts. All raw completions retained. Five seeded samples at a fixed quarter of sites.
- O-lens training placeholder is undocumented. Fixed `<|box_start|>` follows the existing local replication, with independent topic-recovery calibration required before study decoding. This limitation is visible on the dashboard. No substitution of another O-lens occurs silently.
- Paired and unrelated activation controls are references to existing decodes: source text is never fed to the oracle, so only the injected vector changes. No-injection output is retained separately.

## Execution

Remote directory: `/data/agastyas/assistant-jspace-20261008`. Only currently free, confirmed-reserved GPUs may be used. No existing process is stopped or displaced.

`scripts/launch.sh 0` acquires a shared local-user GPU lock, checks for any attached GPU process, and runs capture then oracle sequentially. A reservation confirmation file is required. The launcher must be detached with nohup; results are checkpointed atomically per record/readout. Create `STOP` in the remote directory to stop at the next record/batch boundary. Resume skips completed outputs without changing prompts or selecting outputs.

GPU use has not begun until a live worker and newly saved results are verified. Initial availability: Athena02 GPU 0 empty; Athena02 1–7 occupied; all Athena01 GPUs have attached processes.

## Validation and interpretation

Every factual pair gets a shared-prefix activation check (relative L2 < 0.01; actual errors retained). Every capture verifies block 44 output equals block 45 input exactly, final norm/head parity, and finite activations. O-lens calibration requires expected-topic recovery on at least two of three fixed neutral prompts and an effect of injection. Any failed check stops the stage and writes a failure record.

An assessment that differs from the user's stated stance and follows factual controls is evidence of Assistant-side appraisal; first-person wording is not required. Both perspectives may coexist. This pilot does not establish causal control or a post-training origin. Concept scores and raw lens readouts are not validated perspective classifications. Fresh held-out scenarios are needed for generalization claims.

`scripts/sync_publish.py` only reads this run's remote results, exports an allowlist of study artifacts, and publishes changed dashboard data. It never starts or resumes GPU work. No API keys are needed for the inference or dashboard pipeline.

Source implementations: [J++](https://github.com/safety-research/jpp_lens), [J++ weights](https://huggingface.co/koayon/jpp-lenses), [oracle checkpoint and code](https://huggingface.co/ceselder/oracle-lens-qwen3.6-27b).

Output policy amendment: extra O-lens phrases are accepted by user authorization; every phrase remains visible. Original exact-format flags are preserved as metadata. See `data/oracle-output-policy.json` for qualitative evidence and control failures. This changes acceptance of phrase counts, not prompts, seeds, anchors, or decoding. Resume only the remaining decoder work with `scripts/launch.sh 0 oracle`.
