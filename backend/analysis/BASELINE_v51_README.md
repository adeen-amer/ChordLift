# Baseline v51 — Phase B fine-tune attempt (not shipped)

**Result: does not clear the acceptance bar.** DEV majmin **79.3%** (v50 80.3%, Δ −0.96pp), and the guard track `another-one-bites-the-dust` collapsed to **0.014** (must stay ≥ 0.128 — the threshold exists specifically to protect this track's known tuning-estimator-error case, which v50's 0.40st pitch-candidate cap was built to guard). Per the v50/v51 plan's stop condition, **TEST was not run** and this checkpoint set is **not deployed**. Main stays on v50 (75.0% TEST, merged).

**Winner config attempted:** `--lr 1e-5 --epochs-cap 30 --batch-size 8 --seeds 0,1,2,3,4`, warm-started from the packaged pretrained checkpoints, fine-tuned on 161 train / 46 val (audio, .lab) pairs — Isophonics Beatles+Queen non-gold tracks plus a Billboard-sourced validation set, both leakage-verified against the 24-track gold holdout (`scripts/verify_training_leakage.py`, zero overlap).

## The debugging saga — three false failures, one real bug

Three consecutive full training runs looked catastrophically broken (DEV majmin collapsing to ~0.26, then chance-level confidence ~0.015–0.017 on real audio vs. the pretrained checkpoint's 0.94) before the actual cause was found. Documenting this in detail because the eventual root cause is a landmine for anyone using `CHORD_CHORDIA_MODELS`/`CHORD_CHORDIA_MODEL_DIR` in the future.

**Headline bug — `chord_engine_chordia.py`'s `_model_names()` silently loaded random weights.** Real checkpoint filenames embed a literal comma in their `reweight(0.0,10.0)` suffix (e.g. `joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_ft1_s0.best`). The old `_model_names()` did `raw.split(",")` on `CHORD_CHORDIA_MODELS`, which shredded every fine-tuned model name into two garbage fragments (`...reweight(0.0` and `10.0)_ft1_s0.best`) that matched no file on disk. `NetworkInterface` found no matching checkpoint, `finalized` stayed `False`, and inference silently ran on the network's **random initialization** — no exception, no warning, just output that looked exactly like a broken model (~1/73 chance-level confidence per class). Every prior "root cause" investigation this session (learning-rate-driven catastrophic forgetting, out-of-range triad labels reaching `F.cross_entropy` uncapped) was chasing symptoms of a harness bug, not the training pipeline. Fixed with a paren-aware comma splitter (`_split_model_names`); regression test added (`test_model_list_env_override_preserves_commas_in_parens`). **This bug also breaks `rebaseline_v51.py --models`, and would break production serving if `CHORD_CHORDIA_MODELS` were ever set to a fine-tuned checkpoint name** — it is not Phase-B-specific.

**Secondary fixes kept (real, but not the cause of the observed collapse):**
- `dataset.py`'s `encode_labels()` now calls `complex_chord_chop()` before caching each label. Billboard's `:1` (bare-root/power-chord) annotations encode a triad index of 73–96, exceeding the network's 73-class triad head (measured: 371 such lines across the real corpus). The loss function's `conditional_classifier_loss` only masks `< 0`, never checks the upper bound, so uncapped values reached `F.cross_entropy` directly — a known source of silent GPU gradient corruption rather than a clean crash. Confirmed via diagnostic: frame-level `X`-masking on real data is only ~1.1%, ruling out over-masking as a concern.
- `--lr` lowered from the RUNBOOK's `1e-4` to `1e-5`. Diagnostic testing (single-batch confidence tracking against a frozen reference clip) showed real per-step confidence erosion during fine-tuning at 1e-4 that didn't show up in the loss curve; 1e-5 measurably reduced it. Once the `_model_names()` fix was in place, `1e-4` was never re-tested standalone, so it's possible the LR change was unnecessary — treat `1e-5` as the validated value, not a confirmed requirement.
- `scripts/download_gold_spotdl.py` and `chord_training/fetch_train_data.py` picked up three unrelated Windows-portability fixes this session (`sys.executable` instead of hardcoded `.venv/bin/python`, explicit UTF-8 `subprocess` encoding, and dropping spotdl's `azlyrics` lyrics provider — its `requests.Session.get()` calls carry no timeout and hang forever on an unreachable host). Also `tests/fixtures/gold_youtube_overrides.json`: two dead YouTube override links were replaced with verified working ones (`dont-stop-me-now`, `seven-seas-of-rhye`).

**RUNBOOK.md doc bug:** its "Copy back" step names `*_ft1_s*.best.sdict` as the val-loss-best checkpoint. Per `train.py`'s actual save logic, since `save_name` already ends in `.best` (matching the pretrained warm-start naming), the true best-val file is the double-suffixed `*_ft1_s*.best.best.sdict` — `*.best.sdict` (single suffix) is the end-of-schedule *final* checkpoint. `FINDINGS.md` documents this correctly (§9); the RUNBOOK's copy-back instructions did not match it. Fixed in this pass.

## DEV results (14 tracks, all gold-audio ear-check approved via Spotify official-duration match)

| track | v50 | v51_ft1 | Δ |
|---|---|---|---|
| another-one-bites-the-dust | 0.128 | **0.014** | **−0.115** |
| youre-my-best-friend | 0.834 | 0.794 | −0.040 |
| i-saw-her-standing-there | 0.933 | 0.904 | −0.029 |
| play-the-game | 0.575 | 0.564 | −0.011 |
| something | 0.782 | 0.773 | −0.009 |
| cant-buy-me-love | 0.927 | 0.922 | −0.006 |
| twist-and-shout | 0.940 | 0.934 | −0.006 |
| help | 0.969 | 0.964 | −0.006 |
| ticket-to-ride | 0.961 | 0.955 | −0.005 |
| let-it-be | 0.874 | 0.872 | −0.002 |
| come-together | 0.954 | 0.957 | +0.003 |
| dont-stop-me-now | 0.806 | 0.815 | +0.009 |
| while-my-guitar-gently-weeps | 0.806 | 0.839 | +0.033 |
| yellow-submarine | 0.748 | 0.797 | +0.049 |

**GOLD AGGREGATE (14 tracks):** root WCSR 0.802, majmin 0.793, sevenths 0.668, seg 0.814.

Excluding the guard track, the fine-tune is close to a wash on DEV: small mixed deltas in both directions (−0.04 to +0.05), no systemic regression. The failure is concentrated entirely on `another-one-bites-the-dust`, whose root/majmin/sevenths all collapsed near zero (`ref=105 est=47` segments) while `seg=0.567` stayed moderate — consistent with the decoder finding few confident segments rather than a wholesale extraction failure.

**Diagnosed, not the pitch-selection cap.** `another-one-bites-the-dust`'s known 0.45st tuning-estimator-error candidate is computed by `pitch_utils.estimate_pitch_shift_semitones` (pure signal processing, no chordia model involved) and gets rejected outright by `CHORD_PITCH_MAX_CANDIDATE_SHIFT=0.40` regardless of which checkpoint is loaded — confirmed both v50 and v51_ft1 are scored on the *same* uncorrected audio for this track, ruling out my first hypothesis (confidence-driven branch selection). Direct A/B on the raw stem: the fine-tuned ensemble's mean decoder confidence on this track is actually **higher** than the pretrained model's (0.538 vs 0.487) — so it isn't undertrained or uncertain, it is *confidently wrong*. This points to a genuine generalization gap: the 161-track fine-tuning corpus (mostly Beatles + a Billboard mix skewed toward guitar/vocal-driven pop-rock) underrepresents this track's bass-and-drums-led funk/disco harmony, and fine-tuning shifted the decision boundary away from what this specific pattern needs, even while improving or holding flat on 13/14 other tracks. This is a data-coverage limitation of the current corpus, not a code bug — the fix, if pursued, is a larger and more genre-diverse fine-tuning set (more bass-driven funk/disco/dance tracks), not a hyperparameter tweak.

## Serving config (not applied — v50 remains the shipped baseline)

Not deployed. `backend/chord_training/checkpoints_ft1/` retains the five `*.best.best.sdict` seed checkpoints from this run for reference; `CHORD_CHORDIA_MODELS`/`CHORD_CHORDIA_MODEL_DIR` would point at them if a future attempt clears the bar.

**Protocol note:** TEST was never evaluated — the plan's stop condition (no DEV candidate beats v50, or guard track fails) was hit cleanly on the first full run after the harness bug was fixed. No multiplicity concern; this is a single, honest result.
