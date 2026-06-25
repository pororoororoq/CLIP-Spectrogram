# Work Plan: Modernizing & Hardening the CLIP Audio-Deepfake Detector

**Project:** `~/projects/clip/CLIP_application.ipynb`
**Date:** 2026-06-22
**Goal (user-selected):** Publishable benchmark/paper · Compute: one good GPU (Colab Pro / A100-L4 class) · **Priority: methodology robustness first**, then new datasets, then ElevenLabs cross-generator test.

> ⚠️ **Verification note.** This environment had no web access, so dataset sizes, license terms, and download URLs below come from model knowledge (cutoff Jan 2026) and **must be verified against the official source before you cite any number in a paper.** Every dataset row has a `VERIFY:` action. Treat counts as approximate until confirmed.

---

## 1. Requirements Summary

The 2-year-old notebook adapts OpenAI CLIP (ViT-B/32) to audio deepfake detection by rendering audio as mel-spectrogram "images." It trains three variants (frozen CLIP + MultiLevelAdapter + FC head; CLIP fine-tuned against text prompts; a vanilla CNN baseline) on **WaveFake** and **Fake-or-Real (FoR)**, and reports AUC for zero-shot and few-shot transfer.

**Observed results (from notebook outputs):**

| Setting | AUC | Status |
|---|---|---|
| CNN, train acc | 0.969 | in-distribution only |
| CNN zero-shot (cross-dataset) | 0.5026 | ≈ chance |
| CNN few-shot | 0.5026 | **broken** (loss frozen at 1.7984) |
| CLIP zero-shot | 0.5529 | ≈ chance |
| CLIP few-shot | **0.8507** | **likely leakage-inflated** |
| CLIP few-shot (other run) | 0.4892 | **broken** |

The three asks: (A) integrate modern 2025–26 datasets for train/validation; (B) evaluate against real commercial generators like ElevenLabs; (C) make the methodology and evaluation rigorous and complete enough to publish. Per user priority, **C is the foundation** — the current headline result (0.85) cannot be trusted until leakage is removed, so fixing evaluation precedes generating new numbers.

---

## 2. Confirmed Defects (evidence-cited)

These are not hypotheticals; they are in the committed notebook.

| # | Defect | Location | Effect |
|---|---|---|---|
| D1 | **Eval runs on the TRAIN split.** `zero_shot_dataset` and `few_shot_dataset` both load `datasetWF/train`. | `cell 41`, lines ~44–47 | "Zero-shot test" is measured on the eval set's training partition; not a held-out test. |
| D2 | **Zero-shot and few-shot use identical data.** Same root, same files. | `cell 41` | The two conditions aren't independent. |
| D3 | **Support/query leakage.** `support_indices` and `query_indices` are each `random.sample(range(num_samples), …)` from the same pool; correct disjoint split `indices[support_num:]` is commented out. | `cell 51`, lines ~28–31 | Query set overlaps support set → **0.85 few-shot AUC is inflated**. |
| D4 | **CNN few-shot doesn't learn.** Loss == 1.7984 every epoch; AUC 0.5026. Optimizer/param-group not wired to trainable params (or LR/grad path broken). | `cell 44` | That reported number is meaningless. |
| D5 | **Spectrogram→CLIP domain mismatch.** 1-channel 64-mel → resized 224×224 → normalized with CLIP's *photographic* mean/std; text prompts ("real/fake audio clip") have no visual grounding for spectrograms. | `cells 13, 15, 24, 25` | Explains zero-shot ≈ chance; the pipeline leans entirely on the trained head. |
| D6 | **Shortcut features.** Fixed 25000-sample (~1.5 s @16 kHz) pad/trim → zero-padding silence + duration cues; real-vs-fake corpora may differ in source, creating a domain shortcut rather than artifact detection. | `cell 41` `__getitem__`; `cells 5,7,40` | Model may key on silence/sample-rate/corpus, not deepfake artifacts. |
| D7 | **Thin metrics, no rigor.** Only AUC; single run; no seeds; no confidence intervals; no EER / min-tDCF / a-DCF; no calibration. | all eval cells | Not publishable; field standard is EER + a-DCF with CIs. |

---

## 3. Principles

1. **Trustworthy evaluation before new numbers.** No new dataset or generator is added until splits are leak-free and metrics are field-standard. A wrong 0.85 is worse than an honest 0.70.
2. **Generalization is the research question.** The interesting, publishable claim is cross-generator / cross-corpus robustness, not in-distribution accuracy (which is near-saturated in the field).
3. **Control for shortcuts.** Every reported gain must survive silence-trimming, codec/sample-rate matching, and paired real-vs-fake design — otherwise it's an artifact, not detection.
4. **Reproducibility is a feature.** Fixed seeds, pinned dataset versions/hashes, deterministic splits checked into the repo, scripted (not notebook-only) pipeline.
5. **Honest provenance.** Dataset stats and licenses are verified against primary sources before citation; generated test audio respects each provider's ToS.

## 4. Decision Drivers

1. **Publishability** — reviewers will reject on leakage, missing EER/a-DCF, and single-seed results. These must be fixed first.
2. **Single-GPU budget** — favors frozen-backbone + lightweight head, subsampling, and SSL-feature baselines over full large-model retrains.
3. **Cross-generator validity** — the contribution stands or falls on a clean held-out-generator protocol (incl. ElevenLabs), so the eval harness is the critical path.

## 5. Viable Options (modeling axis)

### Option A — Keep CLIP-on-spectrograms, fix the science around it *(recommended primary)*
**Approach:** Retain the CLIP-image framing as the studied method; rebuild evaluation, splits, metrics, and add modern datasets + ElevenLabs as held-out generators. Frame the paper as "how well does a vision-FM-on-spectrogram detector generalize to modern commercial TTS, and what breaks it."
- **Pros:** Preserves your existing investment and the novel angle; fits single GPU (frozen CLIP + small head); the generalization-gap story is genuinely publishable; honest negative/partial results on commercial generators are valuable to the community.
- **Cons:** CLIP's photographic prior is weak on spectrograms (zero-shot ≈ chance is expected); risk the method underperforms audio-SSL baselines.

### Option B — Add a strong audio-SSL baseline (wav2vec2/WavLM/AASIST) alongside CLIP
**Approach:** Implement an established SSL-front-end + AASIST/linear-head detector as a baseline and compare CLIP against it on the same leak-free splits.
- **Pros:** Reviewers expect a competitive baseline; contextualizes CLIP's numbers; AASIST/wav2vec2 are single-GPU friendly; strengthens the paper regardless of which wins.
- **Cons:** More implementation; if CLIP loses badly, reframing needed (still publishable as a careful comparison).

### Option C — Abandon CLIP, pivot to SSL detector
- **Cons / invalidation:** Throws away the project's distinctive contribution and competes directly with a crowded SOTA field where beating AASIST/wav2vec2 on a single GPU is hard. **Rejected** as the primary path; folded in as the *baseline* (Option B) instead.

**Chosen synthesis: A + B.** CLIP-on-spectrograms is the studied method; an SSL/AASIST detector is the required baseline; both evaluated on one leak-free, codec-controlled, cross-generator protocol. This is the configuration most likely to survive review.

---

## 6. Datasets to Add (VERIFY all stats/links before citing)

Ranked shortlist for a single-GPU, generalization-focused study. **Train on one family, test cross-family + cross-generator.**

| Rank | Dataset | Why it matters | `VERIFY:` |
|---|---|---|---|
| 1 | **In-the-Wild** (Müller et al.) | Real-world audio (politicians/celebs), the canonical generalization stress test; small enough for single GPU. | Confirm HF/Zenodo URL, ~31h / speaker count, license (research-only). |
| 2 | **ASVspoof 5** (2024) | Current community benchmark; modern TTS/VC attacks; official train/dev/eval split + **a-DCF/EER** protocol and tooling. | Confirm registration/license, attack list, official metric scripts. |
| 3 | **MLAAD** (Multi-Language Audio Anti-Spoofing) | Many TTS systems × many languages; excellent cross-generator/cross-lingual held-out source. | Confirm version, generator/language list, license, URL. |
| 4 | **SpoofCeleb** | Large in-the-wild TTS/VC over VoxCeleb-style speakers; speaker-diverse. | Confirm size, split, license, URL. |
| 5 | **CodecFake** / codec-based sets | Targets neural-codec & modern vocoder artifacts that older WaveFake misses; directly relevant to ElevenLabs-era audio. | Confirm exact dataset name/variant, URL, license. |
| 6 (keep) | **WaveFake**, **FoR** | Retain as *legacy/in-distribution* reference points for backward comparison only. | Already in use. |

**Standard protocol to adopt (`VERIFY:` against primary sources):**
- **Canonical training source:** train+tune on **ASVspoof 2019 LA train/dev** (license-clean, attack-diverse, fixed protocol), optionally augmented with ASVspoof 5 train / MLAAD for architecture+language diversity. Keep speaker- *and* generator-disjoint train/dev.
- **Zero-shot hold-outs (never trained or tuned on):** In-the-Wild, MLAAD held-out languages/architectures, WaveFake held-out vocoders, ASVspoof 2021 DF, + the commercial-generator panel (§7). Report EER per target corpus.
- **Leave-one-generator-out (LOGO):** partition spoofed data by producing system; train on all-but-one, test on the held-out generator, rotate — the diagnostic test for "learned a generalizable boundary vs. generator-specific artifacts." Use as a dev signal, not just held-out-attack.
- **Augmentation:** **RawBoost** (convolutive/impulsive/colored noise) + codec/compression/telephony augmentation so the model can't key on clean-synthesis artifacts.
- **Metrics:** primary **EER**; **min-tDCF** for ASVspoof 19/21 LA in-domain, **a-DCF** for ASVspoof 5 (use official scripts); **AUC** secondary; **≥3 seeds**, mean ± 95% CI; pin dataset versions + commit split index files (with content hashes).
- **Founding framing to cite:** Müller et al., *"Does Audio Deepfake Detection Generalize?"* (Interspeech 2022) — introduced In-the-Wild and the in-domain-success ≠ real-world-robustness gap; this is the paper your study extends to the commercial-TTS era. `VERIFY:` arXiv id + exact EER figures.

> **Required baseline (per §5 Option B):** the current SOTA generalization workhorse is an **SSL front-end (wav2vec2 / XLS-R / WavLM) + AASIST**, with RawNet2/AASIST as end-to-end references. Implement wav2vec2-AASIST on the identical splits/metrics; it is what reviewers will compare CLIP against. `VERIFY:` recipe repos before relying on reported numbers.

---

## 7. ElevenLabs / Commercial-Generator Test (held-out generator)

> **Why this is a contribution, not just a sanity check:** to the best of current knowledge, *none* of the standard academic benchmarks (ASVspoof 5, MLAAD, In-the-Wild, SpoofCeleb) include named **commercial-API** generators (ElevenLabs, OpenAI TTS, Cartesia, PlayHT) in their attack rosters — they use open, re-implementable systems. A clean, paired commercial-generator benchmark is therefore a genuine gap your study can fill. `VERIFY:` this gap claim before asserting it in the paper.

**Design = paired, matched, held-out.** This is the cross-generator test, never used in training.

1. **Source real speech** from a held-out real corpus (e.g., LibriSpeech/VCTK test speakers) — speakers/text not seen in training.
2. **Generate cloned/synth counterparts** of the *same transcripts/voices* via ElevenLabs API, plus ≥2 other modern generators (e.g., OpenAI TTS, XTTS/Coqui, Bark, Cartesia, PlayHT) for a generator panel. **`VERIFY:` each provider's ToS permits research/eval use and dataset retention; record provider + model version + date.** Check HuggingFace first for any pre-existing ElevenLabs-generated sets to reduce cost.
3. **Match post-processing across real and fake**: identical sample rate, identical loudness norm, identical silence-trim, and run *both* classes through the same codec battery (WAV / MP3-128 / Opus / "phone" 8 kHz) — so the detector can't key on format. Report per-codec results.
4. **Control for shortcuts**: strip leading/trailing silence (D6); verify with a *silence-only* and *sample-rate-only* probe that the detector isn't separable on those alone.
5. **Report** per-generator EER/AUC, a "seen-generators vs ElevenLabs(unseen)" gap, and at least one qualitative error analysis (Grad-CAM on spectrograms / score histograms).

**Expected, honest hypothesis:** the detector degrades sharply on ElevenLabs (unseen, high-quality, codec-clean) — quantifying that gap *is* the paper's contribution.

---

## 8. Implementation Steps

### Phase 1 — Make evaluation trustworthy (PRIORITY, do first)
1. **Refactor notebook → scripts/modules** (`data/`, `models/`, `eval/`, `configs/`) so runs are reproducible and seed-controlled. Keep a thin notebook for figures only.
2. **Fix D1/D2**: build explicit, disjoint **train/dev/test** index files per dataset; evaluation reads *only* test. Store split files + hashes in repo.
3. **Fix D3**: replace `split_few_shot_loader` with a disjoint support/query split (`indices[:k]` vs `indices[k:]`); add an assertion that `set(support) ∩ set(query) == ∅`.
4. **Fix D4**: repair CNN few-shot — confirm `optimizer` is constructed over the unfrozen params *after* freezing, correct LR, and that `criterion` matches logits/labels; assert loss changes across epochs in a unit test.
5. **Add metric module**: EER, min-tDCF/a-DCF (port ASVspoof 5 scripts), AUC, score calibration; bootstrap 95% CIs; multi-seed runner (≥3 seeds).
6. **Shortcut audits (D6)**: silence-trim all inputs; add silence-only / sample-rate-only / duration-only probe baselines that *should* score ≈ chance — if they don't, a shortcut exists.
7. **Re-run existing WaveFake/FoR experiments** on the leak-free harness → establish corrected baseline numbers (expect the 0.85 to drop). Document the delta.

### Phase 2 — Modern datasets
8. Acquire & `VERIFY:` datasets from §6 (In-the-Wild, ASVspoof 5, MLAAD, SpoofCeleb, CodecFake). Write per-dataset loaders normalizing to a common (waveform, label, meta) schema with provenance fields.
9. Define the **cross-dataset matrix**: train-on-X / test-on-Y for all pairs; report the generalization grid (this table is a core paper figure).

### Phase 3 — Models & baselines
10. Re-run **CLIP variants** (adapter; text-prompt; consider linear-probe on CLIP image features) on the matrix.
11. Implement **SSL baseline** (wav2vec2/WavLM front-end + AASIST or linear head) on the *same* splits/metrics (Option B).
12. Optional ablation: 3-channel spectrogram encodings, mel resolution, ImageNet vs spectrogram-fit normalization (probes D5).

### Phase 4 — Commercial-generator evaluation
13. Build the ElevenLabs + generator-panel paired test set per §7 (with ToS verification + codec battery).
14. Evaluate all models; produce the seen-vs-unseen-generator gap table + error analysis.

### Phase 5 — Write-up & release
15. Results tables (cross-dataset grid, per-generator, per-codec), with CIs; ablations; reproducibility appendix (seeds, versions, hashes); code + split-file release.

---

## 9. Acceptance Criteria (testable)

- [ ] No evaluation reads from any `train` split; an automated test asserts test-set file lists are disjoint from train/dev. *(D1/D2)*
- [ ] `support ∩ query == ∅` assertion passes in the few-shot splitter. *(D3)*
- [ ] CNN few-shot loss strictly decreases across epochs on a smoke test; AUC ≠ 0.5026 fixed. *(D4)*
- [ ] EER, a-DCF/min-tDCF, and AUC reported for every experiment, each as mean ± 95% CI over ≥3 seeds. *(D7)*
- [ ] Silence-only / sample-rate-only probe baselines score within 0.05 AUC of chance on the final test sets (else shortcut documented & mitigated). *(D6)*
- [ ] Corrected WaveFake/FoR numbers reported next to the original (likely-inflated) ones, with the delta explained.
- [ ] ≥4 modern datasets from §6 integrated with verified provenance (URL + license + version recorded in repo).
- [ ] Cross-dataset train/test generalization grid produced.
- [ ] ElevenLabs + ≥2 other generators evaluated as held-out, with matched post-processing and per-codec breakdown; ToS compliance recorded per provider.
- [ ] One competitive SSL baseline (AASIST/wav2vec2) reported on identical splits/metrics.
- [ ] Every cited dataset statistic traced to a primary source (no unverified counts in the paper).

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| CLIP method underperforms SSL baselines | Reframe as a careful generalization study + negative result; Option B baseline makes the paper valuable either way. |
| Dataset access gated (registration/license) | Start access requests early (Phase 2); In-the-Wild + MLAAD are typically lighter-weight; have CodecFake/SpoofCeleb as alternates. |
| ElevenLabs ToS / cost limits generation | Verify ToS first; check HF for pre-generated sets; cap voices/utterances; use the generator panel so no single provider is load-bearing. |
| Single GPU can't fit full retrains | Freeze backbones, subsample, use linear probes; SSL features cached to disk. |
| Codec/silence shortcuts re-enter via new data | Probe baselines (AC §5) run on *every* final test set, not just once. |
| Unverified dataset stats slip into paper | `VERIFY:` gate in AC; provenance fields mandatory in loaders. |

## 11. Verification Steps

1. Run the disjointness + smoke tests (AC items D1–D4) in CI before any experiment.
2. Reproduce corrected WaveFake/FoR baseline twice with different seeds; confirm CI overlap.
3. Run probe baselines on each final test set; archive results.
4. Independent re-run of the cross-dataset grid from scripts (not the notebook) to confirm reproducibility.
5. Cross-check every dataset stat/license against the primary source; record URL + access date.

---

## 12. ADR

- **Decision:** Keep CLIP-on-spectrograms as the studied method, add a required SSL baseline, and rebuild evaluation (leak-free splits, EER/a-DCF, multi-seed, shortcut probes) + modern datasets + held-out commercial-generator (ElevenLabs) test. Methodology fixes ship before any new headline numbers.
- **Drivers:** publishability, single-GPU budget, cross-generator validity.
- **Alternatives considered:** (B) SSL baseline only — adopted as baseline, not replacement; (C) full pivot to SSL — rejected (discards the project's distinctive angle, crowded SOTA).
- **Why chosen:** preserves existing investment, turns the known weakness (poor generalization) into the research contribution, and fits the compute budget.
- **Consequences:** the current 0.85 will likely drop once leakage is fixed — this is expected and must be reported honestly; more eval/infra work up front.
- **Follow-ups:** confirm dataset licenses; decide target venue (e.g., an audio/security workshop) to fix metric conventions; scope how many generators in the panel.

---

## Changelog
- v1 (2026-06-22): Initial plan. Web research for dataset stats/links was blocked in this environment; all dataset specifics carry `VERIFY:` gates pending primary-source confirmation.
