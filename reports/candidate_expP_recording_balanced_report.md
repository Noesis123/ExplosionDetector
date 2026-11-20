# Candidate ExpP: Recording-Balanced Optimization & Source-Inventory Evaluation Report

**Date:** 2026-11-18  
**Project:** `D:\NOESIS\Explosion\Explosion`  
**Active Production Baseline:** `candidate_expF_accuracy_optimized_best.pt` (ExpF + AEV Config C)  
**Evaluated Candidate:** `candidate_expP_recording_balanced_final.pt` (ExpP)  
**Decision:** **REJECT ExpP; PRESERVE ExpF IN PRODUCTION**

---

## Executive Summary

Following the discovery that the verified ExpF training inventory comprises 634 recordings (207 explosions, 427 non-targets) with severe acoustic source deficits (**only 4 unique artillery recordings** across 2 BBC tracks and **13 backfire recordings**), we designed and executed **Candidate ExpP**. 

ExpP implemented **recording-balanced (file-level) pooling** to eliminate window repetition overemphasis (capping each recording to 1 representative window, reducing 144 artillery windows to 4 and 424 backfire windows to 13), combined with quadratic positive-margin anchoring and hard-negative suppression.

While ExpP successfully reduced historical false-positive files from **19/68 down to 15/68** (-4 false alarms in gunfire and artillery), it **failed the strict recall guardrail**:
- **Explosion Recall dropped from 22/24 (91.67%) down to 19/24 (79.17%)**.
- **Three genuine explosions were newly missed**: `bbc_demolition_07022508.mp3` ($0.7022 \rightarrow 0.6602$), `fs_367622_fxkid2.mp3` ($0.7136 \rightarrow 0.6811$), and `fs_323316_alfrodou.mp3` ($0.7205 \rightarrow 0.6697$).
- On the held-out source-isolated validation set (190 files), recall dropped from **84.3% (59/70) down to 75.7% (53/70)** (6 genuine explosions lost).

Under the explicit mandate to **reject any candidate that loses a previously detected explosion**, Candidate ExpP is **categorically rejected**. `candidate_expF_accuracy_optimized_best.pt` remains the verified operational baseline.

---

## 1. Verified Inventory Audit & Source Diversity Findings

Inspection of `training/data/train_fit_binary_pool.pt` established the exact source inventory:

| Split | Unique Recordings | Total Windows | Avg Windows/File | Source Diversity Assessment |
|---|---:|---:|---:|---|
| **Explosions (Positives)** | 207 | 4,094 | 19.8 | Adequate diversity across close, distant, and demolition blasts. |
| **Non-Targets (Negatives)** | 427 | 4,693 | 11.0 | Severe category-level source imbalances. |
| **Total Inventory** | **634** | **8,787** | **13.9** | **Verified ExpF Training Set** |

### Critical Source Diversity Imbalances in Confusing Categories:
1. **Artillery Firing (Only 4 unique recordings, 144 windows — 36.0 win/file):**
   - `bbc_gunfire---_07034186.mp3` (`group_bbc_ec164a_heavy_artillery_shells`)
   - `bbc_guns--gunf_07027149.mp3` (`group_bbc_ec1g1_artillery_25pounder`)
   - `bbc_guns--gunf_07027150.mp3` (`group_bbc_ec1g1_artillery_25pounder`)
   - `bbc_guns--gunf_07027151.mp3` (`group_bbc_ec1g1_artillery_25pounder`)
   Three of the four training artillery files originate from the exact same 25-pounder studio recording. Across the entire training set, there are **only two physical acoustic sound sources** for artillery. In contrast, the Historical Benchmark contains 5 artillery false alarms originating from distinct Freesound field recordings (`fs_171475`, `fs_184275`, `fs_187542`, `fs_184650`, `fs_161341`).

2. **Vehicle Backfires (Only 13 unique recordings, 424 windows — 32.6 win/file):**
   - 12 of the 13 files already produce $p_{\text{exp}} < 0.67$ under ExpF.
   - Only 1 file (`bbc_gas--north_07072079.mp3`, an industrial gas/plant flare) was misclassified, but contributed 35 separate false-positive windows ($p_{\text{exp}} \ge 0.70$)! Naive window sampling caused this single recording to dominate negative batches.

3. **Thunder (23 unique recordings, 276 windows):**
   - 11 files misclassified by ExpF ($p_{\text{exp}} \ge 0.70$).
4. **Gunfire (51 unique recordings, 476 windows):**
   - 15 files misclassified by ExpF ($p_{\text{exp}} \ge 0.70$).

---

## 2. Experimental Optimization: Candidate ExpP Design

To address the findings without introducing new data or altering held-out partitions:
1. **Recording-Balanced Extraction:** Capped every recording to 1 representative peak window (325 positive windows from 207 files; 427 negative windows from 427 files). No single recording could supply more than 1 window per step.
2. **Trainable Capacity:** Frozen Block 1, Block 2, Block 3, and all BatchNorm layers. Only Block 4 Conv2d and the Linear Classifier Head were trained (73,986 parameters).
3. **Loss Function:**
   $$\mathcal{L} = \mathcal{L}_{\text{CE}} + 1.0 \cdot \mathcal{L}_{\text{pos\_margin}} + 1.5 \cdot \mathcal{L}_{\text{neg\_margin}} + 2.0\text{e-3} \cdot \|\theta - \theta_{\text{ExpF}}\|^2$$
   - $\mathcal{L}_{\text{pos\_margin}} = \text{ReLU}(1.15 - (z_{\text{exp}} - z_{\text{nt}}))^2$ on fragile positives ($p_{\text{exp}} \le 0.78$).
   - $\mathcal{L}_{\text{neg\_margin}} = \text{ReLU}((z_{\text{exp}} - z_{\text{nt}}) - (-0.40))^2$ on hard negatives ($p_{\text{exp}} \ge 0.50$).
4. **Training Schedule:** 4 epochs, 50 steps per epoch, $\eta = 3.0\text{e-6}$ with AdamW. Full evaluation after every epoch.

---

## 3. Comprehensive Benchmark Results: ExpF vs ExpP

| Evaluation Suite | Metric | ExpF (Baseline) | Candidate ExpP | Delta |
|---|---|---:|---:|---:|
| **Historical Benchmark (92 files)** | **Explosion Recall** | **22/24 (91.67%)** | **19/24 (79.17%)** | **-12.50% (3 lost)** |
| *(AEV Config C)* | **False-Positive Files** | **19/68 (27.94%)** | **15/68 (22.06%)** | **-4 files (-5.88%)** |
| | True Positives (TP) | 22 | 19 | -3 |
| | False Positives (FP) | 19 | 15 | -4 |
| | True Negatives (TN) | 49 | 53 | +4 |
| | False Negatives (FN) | 2 | 5 | +3 |
| | Precision | 53.66% | 55.88% | +2.22% |
| | F1-Score | **0.6769** | **0.6552** | **-0.0217** |
| | Overall Accuracy | 77.17% | 78.26% | +1.09% |
| **Source-Isolated Val (190 files)** | **Explosion Recall** | **59/70 (84.29%)** | **53/70 (75.71%)** | **-8.58% (6 lost)** |
| *(Raw CNN Dev Set)* | False-Positive Files | 29/120 (24.17%) | 26/120 (21.67%) | -3 files (-2.50%) |
| | Precision | 67.05% | 67.09% | +0.04% |
| | F1-Score | **0.7468** | **0.7114** | **-0.0354** |
| | Overall Accuracy | 78.95% | 77.89% | -1.06% |

---

## 4. Exact File-Level Differences

### Newly Missed Genuine Explosions (3 Lost Files)
The 3 genuine explosions lost by ExpP are precisely the boundary blasts located near the $0.70$ threshold under ExpF:

| Audio Filename | Category | ExpF Max Conf | ExpP Max Conf | Delta | Status Under ExpP |
|---|---|---:|---:|---:|---|
| `bbc_demolition_07022508.mp3` | controlled_demolition | **0.7022** | **0.6602** | -0.0420 | **LOST / MISSED** |
| `fs_367622_fxkid2.mp3` | positives (distant blast) | **0.7136** | **0.6811** | -0.0325 | **LOST / MISSED** |
| `fs_323316_alfrodou.mp3` | positives (quarry blast) | **0.7205** | **0.6697** | -0.0507 | **LOST / MISSED** |

### False Positives Successfully Suppressed (4 Files)

| Audio Filename | Category | ExpF Max Conf | ExpP Max Conf | Delta | Status Under ExpP |
|---|---|---:|---:|---:|---|
| `fs_161341_qubodup.mp3` | artillery_firing | 0.7355 | 0.6982 | -0.0374 | **SUPPRESSED** (Passed AEV) |
| `fs_187542_qubodup.mp3` | artillery_firing | 0.7453 | 0.7081 | -0.0372 | **SUPPRESSED** (Blocked by AEV) |
| `fs_213925_Diboz.mp3` | gunfire | 0.7377 | 0.6891 | -0.0487 | **SUPPRESSED** (Passed AEV) |
| `fs_427597_michorvath.mp3` | gunfire | 0.7268 | 0.6947 | -0.0321 | **SUPPRESSED** (Passed AEV) |

### Remaining False-Positive Files by Category (15 Files Under ExpP):
- `artillery_firing` (3 files): `fs_171475` (0.9321), `fs_184275` (0.7812), `fs_184650` (0.7145)
- `gunfire` (2 files): `bbc_gunfire---_07034194` (0.8012), `bbc_gunfire---_07034201` (0.7714)
- `vehicle_backfires` (3 files): `fs_350672` (0.8412), `bbc_transport-_07012036` (0.8241), `fs_105351` (0.7612)
- `thunder` (2 files): `fs_195529` (0.8612), `fs_36044` (0.8510)
- `bioacoustics` (1 file): `bbc_right-whal_nhu0501615` (0.7912)
- `fireworks` (1 file): `fs_410347` (0.8315)
- `shouting_screaming` (1 file): `p19_shouting_screaming_3` (0.7610)
- `silence_ambience` (1 file): `p19_silence_ambience_4` (0.9012)
- `household_ambience` (1 file): `p19_household_ambience_3` (0.7214)

---

## 5. Root-Cause Analysis: The Inherent Bottleneck

The comparison between lost positives and suppressed false positives reveals an exact mathematical insight:

$$\Delta p \approx -0.035 \pm 0.015 \quad \text{across ALL low-frequency impulsive events}$$

1. **Feature Subspace Overlap:** Because the 2D-CNN uses Mel spectrogram representations, the spectral profile of distant low-frequency artillery and reverberant quarry demolitions map to nearly identical feature activations in Block 4.
2. **Acoustic Asymmetry in Data:** The training inventory contains **only two BBC recordings of artillery**, meaning the model never saw the reverberation profiles of the Freesound artillery negative benchmark files during pre-training.
3. **Linear Margin Shift:** Without separate acoustic features (e.g., rise-time, onset kurtosis, or multiscale temporal envelope tracking), any gradient that pushes down artillery logits exerts an identical downward force on boundary explosions.
4. **Conclusion:** Hard-negative fine-tuning within the existing `ExplosionBinaryCNN` parameter space cannot simultaneously reject artillery/gunfire confounds and retain `bbc_demolition_07022508.mp3`.

---

## 6. Decision & Recommended Next Steps

1. **Maintain ExpF in Production:** `candidate_expF_accuracy_optimized_best.pt` remains the optimal checkpoint on the Pareto frontier (Recall 91.67%, F1 0.6769). Production server daemon (`task-11239` on port 8767) is untouched and verified.
2. **Reject ExpP Checkpoint:** Do not promote `candidate_expP_recording_balanced_final.pt`.
3. **Future Path for FP Reduction:** To achieve $<10$ false positives without sacrificing demolition 08, future work must focus on **post-CNN temporal/kinematic verification in the AEV tracker** (e.g., onset rise-time slopes $< 15\text{ ms}$ and multi-window envelope decay ratios) rather than retraining the CNN on an incomplete 4-recording artillery dataset.
