# CLIP-Spectrogram

Adapting OpenAI CLIP (ViT-B/32) to audio-deepfake detection by rendering audio as
mel-spectrogram "images."

## Notebooks

| File | Purpose |
|---|---|
| `CLIP_application.ipynb` | Original research notebook (2 years old) — kept for reference. |
| `CLIP_application_phase1.ipynb` | **Phase 1: trustworthy evaluation.** Run on Colab (GPU). |
| `CLIP_application_phase2.ipynb` | **Phase 2: modern datasets + cross-dataset generalization grid.** Builds on the Phase 1 core. |

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

## Phase 2 — modern datasets & generalization grid (`CLIP_application_phase2.ipynb`)

Implements Phase 2 of the plan (§8: modern datasets + the cross-dataset matrix). It reuses the
verified Phase 1 core (splits, preprocessing, models, metrics, training, eval) and adds:

- **One `(path, label, meta)` schema with provenance** (source URL · license · version · `verified`
  flag) for every corpus.
- **Dataset adapters**: HF loaders that run on Colab with no token — WaveFake
  (`ajaykarthick/wavefake-audio`), CodecFake (`ajaykarthick/codecfake-audio`), In-the-Wild
  (`mueller91/In-The-Wild`) — plus path-based loaders for MLAAD (`mueller91/MLAAD`),
  ASVspoof 2019 LA, ASVspoof 5, and FoR. A registry skips datasets that aren't available rather than
  failing, so the grid always runs over what you have.
- **Group-disjoint splits** so a *speaker* (In-the-Wild) or *generator* (MLAAD/ASVspoof) never spans
  train and test — the leakage that actually matters for a generalization claim.
- **The cross-dataset generalization grid** (train-on-X / test-on-Y for all pairs) — the core paper
  figure — and **leave-one-generator-out (LOGO)** scaffolding.
- **min-tDCF** wired for ASVspoof LA and an **a-DCF** pointer to the official ASVspoof 5 eval package.

**Dataset stats and licenses carry `VERIFY:` gates.** Identifiers were gathered from search while
direct dataset-card access was blocked; each adapter records provenance with `verified=False` until
you confirm it against the primary source. To run: enable the HF datasets in `P2_ENABLE` (and mount
any gated datasets), then run top-to-bottom; §15 prints the grid, §16 LOGO, §17 the provenance table
and Phase 2 acceptance checks.

> **Verified-so-far:** MLAAD is CC-BY-NC 4.0 (v8+); In-the-Wild ≈58 speakers / ~20.8h real / 17.2h
> fake; ASVspoof 5 uses minDCF/EER + a-DCF via its official eval package. Re-confirm all before citing.
