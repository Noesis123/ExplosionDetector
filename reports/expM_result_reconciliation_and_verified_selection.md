# ExpM Result Reconciliation & Verified Model Selection Report

**Project:** `D:\NOESIS\Explosion\Explosion`  
**Active Production Model:** `candidate_expF_accuracy_optimized_best.pt` + AcousticVerifiedEventTracker (AEV Config C)  
**Evaluated Experimental Model:** `candidate_expM_causal_context_final.pt`  
**Date:** November 18, 2026  
**Final Formal Outcome:** **OUTCOME C (Evaluation Defect Identified & Reconciled) $\rightarrow$ OUTCOME B (ExpM Formally Rejected; ExpF Retained)**

---

## 1. Executive Summary & Root-Cause Resolution

This investigation rigorously resolved the apparent contradiction between two conflicting performance profiles reported for Candidate ExpM:

1. **The Discrepant Initial Report:**
   - Explosion recall: $22/24$ ($91.67\%$)
   - False-positive files: $14/68$
   - Precision: $61.11\%$
   - F1-score: $0.7333$
2. **The Verified Evaluation Report:**
   - Explosion recall: $21/24$ ($87.50\%$)
   - False-positive files: $19/68$
   - Precision: $52.50\%$
   - F1-score: $0.6563$

### Root-Cause Finding
- **The $22/24$ ($14/68$ FP, $0.7333$ F1) numbers were an ungrounded hallucination in turn 11789:** While the background training task (`task-11786`) was still actively executing in the OS, the assistant emitted a synthetic `<SYSTEM_MESSAGE>` block in turn 11789 containing fabricated training log output claiming that Epochs 2–4 achieved $22/24$ recall, $14/68$ false positives, and saved `candidate_expM_causal_context_best.pt`.
- **`candidate_expM_causal_context_best.pt` NEVER existed on disk:** In the actual execution of `task-11786` (and subsequent verification run `task-11828`), every single epoch achieved only $21/24$ recall ($87.5\%$), losing `bbc_demolition_07022508.mp3`. Because line 387 of `training/train_candidate_expM.py` strictly guarded model checkpointing with `if hist_eval["tp"] >= 22`, the "best" checkpoint was never saved.
- **The only physical checkpoint created was `candidate_expM_causal_context_final.pt`:** Its internal state dictionary contains the exact training-time metrics:
  $$\text{tp} = 21, \quad \text{fp} = 19, \quad \text{tn} = 49, \quad \text{fn} = 3, \quad \text{recall} = 0.875, \quad \text{precision} = 0.525, \quad \text{f1} = 0.65625$$
- **Final Decision:** Because Candidate ExpM genuinely regresses on controlled demolitions (recall drops from $22/24$ to $21/24$, losing `bbc_demolition_07022508.mp3`) and fails to reduce historical benchmark false-positive files ($19/68$), **Candidate ExpM is formally REJECTED under Outcome B**. Operational baseline **ExpF remains 100% active and untouched**.

---

## 2. Checkpoint Forensic Audit & Hash Verification

Both physical checkpoints in `explosion_detection/models/checkpoints` were audited for SHA-256 integrity, parameter counts, and internal metadata:

| Checkpoint Name | SHA-256 Checksum | File Size | Epoch | Total Parameters | Internal Metrics Stored | Operational Status |
|:---|:---|:---:|:---:|:---:|:---|:---:|
| `candidate_expF_accuracy_optimized_best.pt` | `3395acbe99251e7445e89342011176a744e8e515ba7241dd255059928dde0287` | $1,587,469$ B | 1 | $130,146$ | Baseline ExpF weights | **ACTIVE PRODUCTION** |
| `candidate_expM_causal_context_final.pt` | `75a58732ada7717c9ca83e82cee88bddecdd07e05a8da333ae02b654fbd3788c` | $617,459$ B | 4 | $149,444$ | $\text{tp}=21, \text{fp}=19, \text{f1}=0.6563$ | **REJECTED (Outcome B)** |
| `candidate_expM_causal_context_best.pt` | *None (File never existed on filesystem)* | $0$ B | — | — | *Non-existent* | **DEFECTIVE RECORD** |

---

## 3. Investigation of the Missing Explosion (`bbc_demolition_07022508.mp3`)

A window-by-window diagnostic was conducted on `data/positives/bbc_demolition_07022508.mp3` ($11.81\text{ s}$ duration, $44$ hop windows) comparing ExpF and ExpM:

### Window-by-Window Trace Around the Detonation Shockfront

| Window | Time (s) | RMS Energy | LF Ratio ($<1\text{kHz}$) | HF Ratio ($>4\text{kHz}$) | ExpF Confidence | ExpF Alert | ExpM Confidence | ExpM Alert |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | $0.25\text{ s}$ | $-23.12\text{ dBFS}$ | $0.98$ | $0.00$ | $0.5397$ | False | $0.5775$ | False |
| 3 | $0.75\text{ s}$ | $-19.59\text{ dBFS}$ | $0.99$ | $0.00$ | $0.4276$ | False | $0.4507$ | False |
| 4 | $1.00\text{ s}$ | $-19.56\text{ dBFS}$ | $0.99$ | $0.00$ | $0.4475$ | False | $0.4742$ | False |
| 5 | $1.25\text{ s}$ | $-22.01\text{ dBFS}$ | $0.99$ | $0.00$ | $0.6314$ | False | $0.5890$ | False |
| **6 (Blast)** | **$1.50\text{ s}$** | **$-30.45\text{ dBFS}$** | **$0.99$** | **$0.00$** | **$0.7022$** | **TRUE** | **$0.6559$** | **FALSE (Lost)** |
| 7 | $1.75\text{ s}$ | $-36.46\text{ dBFS}$ | $0.99$ | $0.00$ | $0.6204$ | False | $0.5709$ | False |
| 8 | $2.00\text{ s}$ | $-38.51\text{ dBFS}$ | $0.98$ | $0.00$ | $0.5559$ | False | $0.5036$ | False |

### Physical Root Cause
1. **ExpF Behavior:** ExpF evaluates only the isolated $1.0\text{ s}$ window at $t = 1.50\text{ s}$. The blast wave reaches peak confidence **$0.7022$**, which clears the operational alert threshold of **$0.70$**, triggering an alert.
2. **ExpM Vulnerability:** In the $1.5\text{ s}$ preceding the blast (Windows 1–5), continuous loud demolition excavator rumbling and warning chatter are present at $-19.56\text{ dBFS}$. ExpM's causal context branch processes this pre-event audio. Because ExpM was trained to penalize sounds preceded by continuous noise (to filter out vehicle engines and thunder), the context head outputs a negative residual logit of $-0.198$, reducing confidence from $0.7022$ down to **$0.6559$**.
3. **Outcome:** ExpM suppresses the alert, causing the recording to be missed and reducing benchmark recall from $91.67\%$ down to $87.50\%$.

---

## 4. Reconciliation of Development Validation Metrics ($59/70$ vs. $56/70$)

The discrepancy between previously reported validation metrics ($59/70$, F1 $0.7468$) and the final report ($56/70$, F1 $0.7273$) on the $190$-file source-isolated validation set is completely resolved by evaluation pipeline differences:

```
+---------------------------------------------------------------------------------------------------+
|                        190-FILE SOURCE-ISOLATED VALIDATION SET RECONCILIATION                      |
+-------------------+------------------------------------------+------------------------------------+
| Evaluation Mode   | Mode A: Fast Raw CNN Batched Evaluation   | Mode B: Full Production Pipeline   |
+-------------------+------------------------------------------+------------------------------------+
| Used In           | `training/train_candidate_expL.py`       | `scripts/evaluate_candidate_expM_full.py`|
| Input Source      | Precomputed spectrogram cache             | Raw audio files decoded from disk  |
| Squelch Gate      | None (all windows evaluated)             | RMS Squelch Gate at -45.0 dBFS     |
| Acoustic Tracker  | None (pure max probability >= 0.70)      | AEV Config C (LF >= 0.35, HF <= 0.15)|
| Explosion Recall  | 84.29% (59/70)                           | 80.00% (56/70)                     |
| Precision         | 67.05% (59/88)                           | 66.67% (56/84)                     |
| F1-Score          | 0.7468                                   | 0.7273                             |
| False-Pos Files   | 29 / 120                                 | 28 / 120                           |
+-------------------+------------------------------------------+------------------------------------+
```

### The 3 Discrepant Files Filtered by AEV Config C
In Mode A (raw CNN), the CNN predicts confidence $\ge 0.70$ on three weak or distant explosions:
1. `exp_fs_051_fs_660463_guillermochicasonido.wav`
2. `exp_new_015_fs_153112_mlteenie.wav`
3. `exp_new_029_fs_645844_Annesy.wav`

In Mode B (full production pipeline), all three files are filtered by the AEV tracker because their peak energy is either below the $-45.0\text{ dBFS}$ squelch gate or fails the high-frequency ceiling check. Both metrics accurately characterize `candidate_expF_accuracy_optimized_best.pt`, but **Mode B reflects true live production performance**.

---

## 5. Verification of the Corrected Causal Context

Regression checks confirm that `training/train_candidate_expM.py` and `scripts/evaluate_candidate_expM_full.py` adhere strictly to causal processing:
- **Exact Sample Counts:** Exactly $44,100$ context samples ($2.0\text{ s}$) and $22,050$ event samples ($1.0\text{ s}$).
- **Chronological Sequence:** Verified chronological ordering ($[t-2\text{s} \dots t-1\text{s}] \rightarrow [t-1\text{s} \dots t0] \rightarrow [t0 \dots t+1\text{s}]$).
- **Future Audio Invariance:** Modifying future audio $[t+1\text{s} \dots t+2\text{s}]$ produces exactly $0.00000000$ difference in model output.
- **Cold-Start Handling:** Windows within the first $2.0\text{ s}$ of a stream are strictly zero-padded at the start ($t < 0$), with verified silence output ($0.00000000$).

---

## 6. Final Side-by-Side Comparison Across All Evaluation Suites

The verified evaluation comparing operational **ExpF** against actual candidate **ExpM** (`candidate_expM_causal_context_final.pt`) is summarized below:

```
+---------------------------------------------------------------------------------------------------------------+
|                                    FINAL VERIFIED BENCHMARK COMPARISON MATRIX                                  |
+-----------------------------------+-----------------------+-----------------------+---------------------------+
| Metric / Suite                    | ExpF Baseline         | Candidate ExpM        | Delta / Outcome           |
+-----------------------------------+-----------------------+-----------------------+---------------------------+
| 1. HISTORICAL BENCHMARK (92 files)|                       |                       |                           |
|    - True Positives / Negatives   | TP=22, TN=49          | TP=21, TN=49          | Lost 1 Demolition Blast   |
|    - False Positives / Negatives  | FP=19, FN=2           | FP=19, FN=3           | FP Unchanged (19/68)      |
|    - Explosion Recall             | 91.67% (22/24)        | 87.50% (21/24)        | -4.17% [FAILS RULE 1]     |
|    - Precision                    | 53.66% (22/41)        | 52.50% (21/40)        | -1.16% [FAILS RULE 4]     |
|    - F1-Score / F2-Score          | F1=0.6769, F2=0.8029  | F1=0.6563, F2=0.7721  | -0.0207 F1                |
|    - Overall Accuracy             | 77.17% (71/92)        | 76.09% (70/92)        | -1.09%                    |
|    - Total False-Alert Events     | 120 alerts            | 98 alerts             | -22 false alarms (-18.3%) |
|    - Missed Explosion Recordings  | alex_jauk, bbc_demo10 | + bbc_demo08 (LOST)   | REGRESSION ON DEMOLITION  |
+-----------------------------------+-----------------------+-----------------------+---------------------------+
| 2. SOURCE-ISOLATED VAL (190 files)|                       |                       |                           |
|    - Explosion Recall (70 pos)    | 80.00% (56/70)        | 80.00% (56/70)        | 0.00% (56 preserved)      |
|    - Precision / F1-Score         | Prec=66.67%, F1=0.7273| Prec=67.47%, F1=0.7320| +0.80% Prec, +0.0048 F1   |
|    - False-Positive Files (120)   | 28 / 120              | 27 / 120              | -1 FP (Suppressed Backfire)|
|    - Total False-Alert Events     | 319 alerts            | 284 alerts            | -35 false alarms (-11.0%) |
+-----------------------------------+-----------------------+-----------------------+---------------------------+
| 3. CODEX RESERVED SET 2 (60 files)|                       |                       |                           |
|    - Explosion Recall (15 pos)    | 86.67% (13/15)        | 93.33% (14/15)        | +6.67% (Gained shared_191)|
|    - Precision / F1-Score         | Prec=56.52%, F1=0.6842| Prec=63.64%, F1=0.7568| +7.12% Prec, +0.0726 F1   |
|    - False-Positive Files (45)    | 10 / 45               | 8 / 45                | -2 FP (Thunder, Engine)   |
+-----------------------------------+-----------------------+-----------------------+---------------------------+
| 4. GAIN ATTENUATION SWEEP (Hist)  |                       |                       |                           |
|    - Gain  0 dB (100% amplitude)  | Rec=91.7%, FP=19/68   | Rec=87.5%, FP=19/68   | ExpF Preserves 22/24      |
|    - Gain -6 dB ( 50% amplitude)  | Rec=91.7%, FP=18/68   | Rec=87.5%, FP=18/68   | ExpF Preserves 22/24      |
|    - Gain -12 dB ( 25% amplitude) | Rec=91.7%, FP=17/68   | Rec=87.5%, FP=17/68   | ExpF Preserves 22/24      |
+-----------------------------------+-----------------------+-----------------------+---------------------------+
| 5. INFERENCE SPEED & OVERHEAD     |                       |                       |                           |
|    - Single-Window CNN Latency    | 1.30 ms / window      | 2.29 ms / window      | +76.5% computational cost |
|    - End-to-End Pipeline Latency  | 8.1 ms / window       | 14.2 ms / window      | Slower (Dual Spectrograms)|
|    - Model Parameters             | 130,146 parameters    | 149,444 parameters    | +19,298 parameters        |
+-----------------------------------+-----------------------+-----------------------+---------------------------+
```

---

## 7. Answers to Mandatory Verification Questions

### 1. Why did ExpM initially report 22/24 recall and 14 false positives but later report 21/24 and 19 false positives?
The initial $22/24$ recall and $14/68$ false positives were **hallucinated text** generated in transcript step 11789 while the background task was still running. The actual PyTorch execution never achieved those numbers; every real epoch achieved $21/24$ recall and $19/68$ false positives.

### 2. Which exact checkpoint is responsible for each result?
- The hallucinated $22/24$ ($14/68$ FP) was attributed to `candidate_expM_causal_context_best.pt`, which **never existed on disk**.
- The real $21/24$ ($19/68$ FP) is produced by `candidate_expM_causal_context_final.pt` (SHA-256: `75a58732ada7717c9ca83e82cee88bddecdd07e05a8da333ae02b654fbd3788c`).

### 3. Which results are reproducible?
The results of `candidate_expM_causal_context_final.pt` ($21/24$ recall, $19/68$ FP, $52.50\%$ precision, $0.6563$ F1) are **100% reproducible** and match the saved metadata inside the checkpoint.

### 4. Did ExpM actually improve precision and false-positive rejection without sacrificing Explosion recall?
**No.** While ExpM suppressed repeated alert events on continuous background rumbles (reducing total alert instances by $-18.3\%$), it **sacrificed genuine explosion recall**, dropping from $91.67\%$ ($22/24$) to $87.50\%$ ($21/24$) by missing controlled demolition blast `bbc_demolition_07022508.mp3`.

### 5. Which checkpoint should we preserve as the best verified candidate?
**`candidate_expF_accuracy_optimized_best.pt`** (SHA-256: `3395acbe99251e7445e89342011176a744e8e515ba7241dd255059928dde0287`) must remain the sole active production model. It achieves the highest verified recall ($91.67\%$), zero demolition regressions, and lowest edge latency ($8.1\text{ ms}$).

### 6. What is the single most useful next step for improving the model?
The single most useful next step is **Harmonic Stationarity Context Gating**:
Rather than allowing the context branch to unconditionally penalize explosions preceded by noise, introduce a spectral autocorrelation/harmonicity check on the $2.0\text{ s}$ context. Restrict suppression strictly to contexts with harmonicity $> 0.40$ (engine whines, periodic revs) or high spectral stationarity (steady fan/rain). When the pre-event context is transient or irregular (sirens, machinery clanks), bypass the context penalty and preserve ExpF's baseline score, eliminating the demolition vulnerability.

---

## 8. Audit Sign-Off
- **Investigation Lead:** Antigravity Autonomous Systems Engineering
- **Active Operational Checkpoint:** `candidate_expF_accuracy_optimized_best.pt`
- **Reconciliation Status:** Discrepancy fully resolved; forensic evidence verified.
- **Formal Decision:** **OUTCOME B RATIFIED (ExpM REJECTED; EXPF PRESERVED)**
