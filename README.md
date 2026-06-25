# CLIP-Spectrogram

Adapting OpenAI CLIP (ViT-B/32) to audio-deepfake detection by rendering audio as
mel-spectrogram "images."

## Notebooks

| File | Purpose |
|---|---|
| `CLIP_application.ipynb` | Original research notebook (2 years old) — kept for reference. |
| `CLIP_application_phase1.ipynb` | **Phase 1 rebuild: trustworthy evaluation.** Run this on Google Colab (GPU). |

## Phase 1 — trustworthy evaluation (`CLIP_application_phase1.ipynb`)

Implements Phase 1 of `.omc/plans/clip-deepfake-robustness-plan.md`. The original notebook's
headline **CLIP few-shot AUC = 0.8507** was inflated by evaluation leakage; Phase 1 removes the
leakage and replaces thin metrics with field-standard ones, so the corrected baselines can be
trusted before any new datasets or generators are added.

**Defects fixed**

- **D1/D2** — eval ran on the *train* split → disjoint, deterministic, hashed `train/dev/test/support`
  manifests; evaluation reads **test only**.
- **D3** — few-shot support/query overlapped → disjoint stratified splitter with a
  `support ∩ query == ∅` assertion.
- **D4** — CNN few-shot loss was frozen (optimizer bound to a stale model) → optimizer rebuilt over
  the unfrozen head of the loaded model, with a loss-strictly-decreases assertion.
- **D6** — zero-pad/duration shortcuts → silence-trimming plus duration/silence/sample-rate **probe
  baselines** that must score ≈ chance.
- **D7** — AUC-only, single run → **EER + AUC + calibration + bootstrap 95% CIs over ≥3 seeds**;
  official min-tDCF wired in for Phase 2.

(**D5**, the CLIP photographic-normalization mismatch, is intentionally kept as the studied method;
its ablation is Phase 3.)

**How to run**

1. Open `CLIP_application_phase1.ipynb` in Colab; set Runtime ▸ Change runtime type ▸ **GPU**.
2. Run top-to-bottom. It streams a balanced, deterministic WaveFake subset from the Hugging Face
   Hub (no Kaggle token required). Fake-or-Real is optional for a cross-corpus number.
3. `CFG.smoke_test = True` (default) does a fast end-to-end sanity pass; set it to `False` for the
   real multi-seed numbers.
4. §10 prints corrected-vs-original baselines; §11 runs the acceptance-criteria self-tests.
