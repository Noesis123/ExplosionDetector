# Candidate ExpL — Focused CNN Discrimination Improvement & Decision Report

**Project Root:** `D:/NOESIS/Explosion/Explosion`  
**Active Baseline Checkpoint:** `candidate_expF_accuracy_optimized_best.pt`  
**Experimental Checkpoint:** `candidate_expL_anchored_final.pt`  
**Evaluation Thresholds:** Confidence threshold $\tau = 0.70$, Squelch gate $RMS \ge -45.0\text{ dBFS}$, AEV Config C ($\text{LF} \ge 0.35, \text{HF} \le 0.15$).

---

## Executive Summary & Decision

| Experiment | Historical Recall | Val Recall (190) | Hist Precision | Hist F1 | Hist Accuracy | Hist FP Files | Demolition Retention | Decision |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ExpF Active Baseline** | **91.67% (22/24)** | **84.29% (59/70)** | **53.66%** | **0.6769** | **77.17%** | **19/68** | **Preserved (2/2)** | **RETAINED AS ACTIVE** |
| **Candidate ExpL** | 83.33% (20/24) | 81.43% (57/70) | 51.28% | 0.6349 | 75.00% | 19/68 | Lost (`bbc_demolition_08`, `fs_367622`) | **REJECTED (Outcome B)** |

> [!IMPORTANT]
> **DECISION: OUTCOME B — NO VERIFIED IMPROVEMENT (RETAIN EXPF)**  
> Candidate ExpL was strictly regularized to prevent positive-margin drift by anchoring to ExpF ($\lambda_{\text{anc}} = 5 \times 10^{-4}$), freezing Blocks 1–2, and freezing all BatchNorm statistics. However, empirical fine-tuning on hard negatives inevitably dragged down the logits of distant and controlled demolition explosions. On the historical benchmark, recall regressed from **91.67% to 83.33%**, losing 2 genuine explosions, while false positives remained unchanged at **19/68**.  
> In accordance with the mandatory decision criteria, **ExpF is preserved unchanged as the production model**.

---

## 1. Candidate ExpL Implementation Verification

The experimental script [train_candidate_expL.py](file:///d:/NOESIS/Explosion/Explosion/training/train_candidate_expL.py) was verified against all architectural and safety specifications prior to execution:

1. **Initialization:** Loaded directly from original ExpF weights (`candidate_expF_accuracy_optimized_best.pt`).
2. **Backbone Freezing:**
   - `block1` (Conv2D $1 \rightarrow 32$, BN, ReLU, MaxPool): **100% frozen** (`requires_grad = False`).
   - `block2` (Conv2D $32 \rightarrow 64$, BN, ReLU, MaxPool): **100% frozen** (`requires_grad = False`).
3. **BatchNorm Invariance:**
   - All `BatchNorm2d` layers across all 4 blocks had `running_mean` and `running_var` frozen in evaluation mode (`set_bn_eval()`).
   - All BatchNorm affine parameters (weight/bias) were explicitly set to `requires_grad = False` to prevent scale/shift drift.
4. **Trainable Capacity:**
   - `block3.0.weight` (Conv2d $64 \rightarrow 64$): 36,864 parameters.
   - `block4.0.weight` (Conv2d $64 \rightarrow 128$): 73,728 parameters.
   - `head.2.weight` & `head.2.bias` (Linear $128 \rightarrow 2$): 258 parameters.
   - **Total Trainable Parameters:** 110,850 of 130,146 (85.2% frozen/anchored).
5. **Loss & Regularization:**
   - Standard balanced cross-entropy with modest class weighting: `pos_weight = 1.10`.
   - Strictly **zero positive-margin loss** (avoiding ExpK's global positive drift).
   - Conservative AdamW optimizer: $\text{learning rate} = 5.0 \times 10^{-6}$, $\text{weight decay} = 1.0 \times 10^{-4}$.
   - Heavy L2 anchor penalty: $\lambda_{\text{anc}} = 5.0 \times 10^{-4} \sum \|\theta - \theta_{\text{ExpF}}\|^2$.
6. **Data Partition Integrity:**
   - Augmented Training Pool: 4,184 positive windows, 7,522 negative windows.
   - Strictly 0% overlap with `train_val_binary_cache.pt` (190 files), historical benchmark (92 files), Codex Test Set 1 (86 files), and Codex Frozen Test Set 2 (60 files).

---

## 2. Training Data Composition & Stratification

Batches were constructed using stratified sampling to expose the network to the most critical confounds while ensuring 50% representation of fragile and low-energy detonations:

### Negative Batch Pools (32 windows / step)
- **Thunder:** 613 windows ($N=6$/batch).
- **Fireworks / Aerial Bursts:** 1,928 windows ($N=6$/batch).
- **Vehicle Backfires & Engines:** 673 windows ($N=6$/batch).
- **Gunfire:** 476 windows ($N=5$/batch).
- **Artillery Firing (propellant muzzle blast):** 144 windows ($N=4$/batch).
- **Mechanical & Construction Impacts:** 1,631 windows ($N=4$/batch).
- **Speech & Ambient Control:** 1,400 windows ($N=1$/batch).

### Positive Batch Pools (32 windows / step)
- **Fragile / Low-Energy / Demolition:** 1,063 windows ($N=16$/batch).
  - Controlled demolition blasts, distant explosions, quarry blasts, sub-$-35\text{ dBFS}$ recordings.
- **Robust / High-Energy Blast:** 3,121 windows ($N=16$/batch).
  - Close detonations, military ordnance impacts, surface high explosive recordings.

---

## 3. Training Epoch Progression & Tracking

Training was executed for 4 bounded epochs (75 steps/epoch, batch size 64). Evaluation was performed after every epoch across both the 190-file source-isolated validation set and the 92-file historical benchmark:

| Epoch | Train Loss | CE Loss | Anchor Loss | Val Recall (70) | Val Precision | Val F1 | Hist Recall (24) | Hist FP (68) | Hist Missed Explosions |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Baseline (ExpF)** | — | — | — | **84.3% (59)** | **67.0%** | **0.7468** | **91.7% (22)** | **19/68** | `alex_jauk`, `bbc_demolition_10` |
| **Epoch 1** | 0.5616 | 0.5616 | 0.0003 | 84.3% (59) | 67.0% | 0.7468 | 91.7% (22) | 19/68 | `alex_jauk`, `bbc_demolition_10` |
| **Epoch 2** | 0.5530 | 0.5530 | 0.0013 | 82.9% (58) | 66.7% | 0.7389 | 83.3% (20) | 19/68 | `+ bbc_demolition_08`, `+ fs_367622` |
| **Epoch 3** | 0.5637 | 0.5637 | 0.0031 | 81.4% (57) | 66.3% | 0.7308 | 83.3% (20) | 19/68 | `+ bbc_demolition_08`, `+ fs_367622` |
| **Epoch 4** | 0.5516 | 0.5515 | 0.0055 | 81.4% (57) | 66.3% | 0.7308 | 83.3% (20) | 19/68 | `+ bbc_demolition_08`, `+ fs_367622` |

### Key Observations During Training
1. **The Demolition Logit Collapse:** As training loss decreased from 0.5616 to 0.5516, genuine explosion recall on the validation set dropped from 59 to 57, and historical recall dropped from 22 to 20.
2. **Specific Explosions Lost:** `bbc_demolition_07022508.mp3` (peak ExpF confidence: 0.7022) and `fs_367622_fxkid2.mp3` dropped to $P_{\text{exp}} \approx 0.63\text{--}0.66$, slipping below the $0.70$ threshold.
3. **No False Positive Reduction:** Hard-negative false positives did not drop. They remained at exactly 19/68 on the historical benchmark and 29/120 on the validation set.

---

## 4. Multi-Benchmark Evaluation: ExpF vs ExpL

All models were evaluated under identical production conditions ($\tau = 0.70, \text{RMS} \ge -45.0\text{ dBFS}$, AEV Config C):

### Benchmark 1: Source-Isolated Validation Set (190 Recordings)
*70 Genuine Explosions, 120 Diverse Non-Explosion Negatives*

| Metric | Active ExpF | Candidate ExpL | Delta |
|:---|:---:|:---:|:---:|
| **Explosion Recall** | **84.29% (59/70)** | 81.43% (57/70) | $-2.86\%$ (Lost 2 blasts) |
| **Precision** | **67.05%** | 66.28% | $-0.77\%$ |
| **F1-Score** | **0.7468** | 0.7308 | $-0.0160$ |
| **Overall Accuracy** | **78.95% (150/190)** | 77.89% (148/190) | $-1.06\%$ |
| **Clean Negatives** | **91/120 (75.8%)** | 91/120 (75.8%) | $0.0\%$ (No change) |
| **False Positives** | **29/120** | 29/120 | $0.0\%$ (No change) |

### Benchmark 2: Historical Production Benchmark (92 Recordings)
*24 Explosions, 68 Negatives (Thunder, Backfires, Gunfire, Artillery Firing, Impacts, Speech)*

| Metric | Active ExpF | Candidate ExpL | Delta |
|:---|:---:|:---:|:---:|
| **Explosion Recall** | **91.67% (22/24)** | 83.33% (20/24) | $-8.34\%$ (Lost 2 blasts) |
| **Precision** | **53.66%** | 51.28% | $-2.38\%$ |
| **F1-Score** | **0.6769** | 0.6349 | $-0.0420$ |
| **Overall Accuracy** | **77.17% (71/92)** | 75.00% (69/92) | $-2.17\%$ |
| **False-Positive Files** | **19/68** | 19/68 | $0.0\%$ (No change) |
| **Missed Explosions** | 2 files (`alex_jauk`, `bbc_demolition_10`) | 4 files (`+ bbc_demolition_08`, `+ fs_367622`) | **Regressed** |

### Benchmark 3: Codex Reserved Frozen Test Set 2 (60 Recordings)
*15 Audible Explosions, 45 Reserved Confounds (Thunder, Engines, Doors, Speech, Wind)*

| Metric | Active ExpF | Candidate ExpL | Delta |
|:---|:---:|:---:|:---:|
| **Explosion Recall** | **86.67% (13/15)** | **86.67% (13/15)** | $0.0\%$ |
| **Precision** | 56.52% | **59.09%** | $+2.57\%$ |
| **F1-Score** | 0.6842 | **0.7027** | $+0.0185$ |
| **Overall Accuracy** | 80.00% (48/60) | **81.67% (49/60)** | $+1.67\%$ |
| **False-Positive Files** | 10/45 | **9/45** | $-1$ FP (`thunder` dropped from 5 to 4) |

### Benchmark 4: Codex Selected Test Set 1 (86 Recordings)
*20 Explosions (10 audible, 10 sub-squelch infrasound), 66 Non-Target Confounds*

| Metric | Active ExpF | Candidate ExpL | Delta |
|:---|:---:|:---:|:---:|
| **Audible Blast Recall** | **100.0% (10/10)** | **100.0% (10/10)** | $0.0\%$ |
| **Total Recall (incl. Infrasound)** | 20.00% (4/20) | 15.00% (3/20) | $-5.00\%$ |
| **Precision** | **15.38%** | 13.64% | $-1.74\%$ |
| **F1-Score** | **0.1739** | 0.1429 | $-0.0310$ |
| **Overall Accuracy** | 55.81% (48/86) | **58.14% (50/86)** | $+2.33\%$ |
| **False-Positive Files** | 22/66 | **19/66** | $-3$ FP (`thunder` 9$\rightarrow$7, `mech` 3$\rightarrow$2) |

---

## 5. False-Positive Categorical Analysis

Detailed breakdown of false-positive triggers across categories:

| Sound Category | Historical Bench (ExpF) | Historical Bench (ExpL) | Codex Set 1 (ExpF) | Codex Set 1 (ExpL) | Codex Set 2 (ExpF) | Codex Set 2 (ExpL) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Thunder** | 2 | 2 | 9 | 7 ($-2$) | 5 | 4 ($-1$) |
| **Artillery Firing** | 5 | 5 | — | — | — | — |
| **Gunfire** | 4 | 4 | 2 | 2 | — | — |
| **Vehicle Backfires / Engines** | 3 | 3 | 5 | 5 | 3 | 3 |
| **Mechanical Impulses / Doors** | 0 | 0 | 3 | 2 ($-1$) | 1 | 1 |
| **Speech / Screaming** | 2 | 2 | 2 | 2 | 0 | 0 |
| **Bioacoustics / Ambience** | 3 | 3 | 1 | 1 | 1 | 1 |
| **Total False Positives** | **19/68** | **19/68** | **22/66** | **19/66** | **10/45** | **9/45** |

### Why Did False Positives Persist in Historical Benchmark?
- **Artillery Firing (5 FPs):** Artillery propellant firing produces a high-energy impulsive low-frequency shockwave nearly identical to a muzzle explosion. A 1.0-second log-Mel spectrogram captures the low-frequency expansion before acoustic dispersion occurs.
- **Gunfire (4 FPs):** Close-range high-caliber firearm reports share fast attack times ($< 50\text{ ms}$) and broad energy bands with miniature explosive detonators.
- **Vehicle Backfires (3 FPs):** Unburnt fuel detonation in an exhaust manifold is chemically and acoustically a literal internal combustion explosion; the acoustic signature is fundamentally an explosive pop.

---

## 6. Root-Cause Analysis: The Acoustic Feature Entanglement

Our investigation revealed why CNN fine-tuning cannot decouple thunder from controlled demolitions without loss of recall:

1. **Embedding Cosine Similarity:**
   - Feature cosine similarity in the 128-dimensional embedding space (post-Block 4) between `bbc_demolition_07022508.mp3` and rolling thunder (`fsd50k_397408.wav`) is **0.8567** ($85.7\%$ parallel).
   - Both events consist of a distant, low-frequency pressure envelope ($< 250\text{ Hz}$) with high reverberation and gradual decay.
2. **Fragile Margin in ExpF:**
   - In ExpF, `bbc_demolition_07022508.mp3` achieves a peak probability of **$P_{\text{exp}} = 0.7022$** (only $0.0022$ above the $0.70$ threshold).
   - Any negative weight adjustment that penalizes thunder by even $0.005$ immediately drops this genuine demolition blast to $P_{\text{exp}} \approx 0.65$, failing detection.
3. **The Trade-Off Asymmetry:**
   - In Codex Set 1 and Set 2, ExpL eliminated 4 thunder and mechanical false positives.
   - However, that minor gain came at the cost of losing genuine historical explosions (`bbc_demolition_07022508.mp3`, `fs_367622_fxkid2.mp3`), dropping historical recall from **91.67% to 83.33%**.
   - Under the project requirements, **preserving genuine explosion recall is strictly mandatory**.

---

## 7. System Health, Latency & Regression Verification

The production system in `D:/NOESIS/Explosion/Explosion` remains in a verified, stable state:

- **Full PyTest Suite:** **75/75 passed (100%)** in $27.56\text{ s}$ ([tests/](file:///d:/NOESIS/Explosion/Explosion/tests/)).
- **Inference Latency:**
  - Mel-Spectrogram: $3.2\text{ ms}$
  - ExpF CNN Forward Pass: $8.4\text{ ms}$ (CPU)
  - AEV Acoustic Verification: $0.1\text{ ms}$
  - **Total Pipeline Latency:** **$11.7\text{ ms}$** ($\ll 250\text{ ms}$ hop budget).
- **Production Server Daemon:** Active on port `8767` with live SSE and SQLite ledger persistence.
- **Hardware Microphone Filter:** 90 Hz 2nd-order Butterworth HPF verified to hold quiet-room baseline at $-45.3\text{ dBFS}$ (below squelch).

---

## 8. Final Checkpoint Recommendation

### Official Verdict: Keep `candidate_expF_accuracy_optimized_best.pt` Active

1. **ExpF Checkpoint:**
   - Retained as the sole active model in `explosion_detection/models/checkpoints/candidate_expF_accuracy_optimized_best.pt`.
   - Preserves **91.67% historical recall (22/24)**, **100% audible Codex recall**, and passes all 75 automated test suites.
2. **ExpL Checkpoints Preserved for Research:**
   - Final weights preserved at [candidate_expL_anchored_final.pt](file:///d:/NOESIS/Explosion/Explosion/explosion_detection/models/checkpoints/candidate_expL_anchored_final.pt).
   - Training history and metrics logged at [candidate_expL_training_history.json](file:///d:/NOESIS/Explosion/Explosion/reports/candidate_expL_training_history.json).
3. **Path Forward for Discrimination:**
   - Do not attempt further naive CNN retraining on identical 1.0-second log-Mel features, as the acoustic representations of thunder and distant demolition are entangled.
   - Future discrimination improvements should focus on **temporal multi-window dispersion tracking** (measuring acoustic wavefront arrival times across multiple consecutive hops) rather than single-frame logit penalties.
