# MGR840 – Lab 1: Voice Digitization, Voice Activity Detection & Traffic Dimensioning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green)
![XGBoost](https://img.shields.io/badge/XGBoost-regression-red)

Lab work for the course **MGR840** (Master's in Telecommunications Networks, ÉTS Montréal, Fall 2026), completed as a pair.

**Authors:** Ibrahim Kamraoui & Khalil Hanouky

This lab connects classic telephony theory (PCM quantization, Erlang B) with machine learning. It is split into four parts that follow the life of a voice call, from the signal, to the decision of whether to send it, to the network capacity needed to carry it.

---

## Table of contents

1. [Overview](#overview)
2. [Repository structure](#repository-structure)
3. [Getting started](#getting-started)
4. [Part 1 – Quantization: uniform vs µ-law](#part-1--quantization-uniform-vs-µ-law)
5. [Part 2 – Voice Activity Detection with ML](#part-2--voice-activity-detection-with-ml)
6. [Part 3 – Bandwidth savings](#part-3--bandwidth-savings)
7. [Part 4 – Erlang B dimensioning with traffic forecasting](#part-4--erlang-b-dimensioning-with-traffic-forecasting)
8. [Summary of results](#summary-of-results)
9. [Limitations and observations](#limitations-and-observations)
10. [Tech stack](#tech-stack)

---

## Overview

| Part | Topic | Question answered |
|------|-------|-------------------|
| 1 | Quantization (uniform vs µ-law) | Which quantizer gives better quality for loud vs quiet sounds? |
| 2 | Voice Activity Detection (VAD) | Can a model tell speech frames from silence using simple audio features? |
| 3 | Bandwidth savings | How much bandwidth is saved if silence is not transmitted? |
| 4 | Erlang B + XGBoost | How many trunk channels are needed, and can forecasting reduce that? |

---

## Repository structure

```
.
├── MGR840_TP_KAMRAOUI_HANOUKY.ipynb   # Main notebook (all 4 parts)
├── assets/                            # Figures exported from the notebook
│   ├── part1_signals.png
│   ├── part2_confusion_logreg.png
│   ├── part2_confusion_rf.png
│   └── part4_traffic_forecast.png
└── README.md
```

> The datasets are provided by the course and are **not included** in this repository.

Expected data files:

| File | Used in | Content |
|------|---------|---------|
| `vad_dataset_train.csv` | Part 2 | 10,000 labeled audio frames (training) |
| `vad_dataset_test.csv` | Parts 2–3 | 3,000 labeled audio frames (testing) |
| `traffic_24h.csv` | Part 4 | Offered traffic in Erlangs over 24 h, one row per 15 minutes |

---

## Getting started

The notebook was developed on **Google Colab** and reads the datasets from Google Drive.

1. Upload the three CSV files to a Drive folder named `MGR840_LAB1_DATA`.
2. Open the notebook in Colab and run the first cell to mount Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Run the cells in order.

**Running locally instead:** replace the Drive paths with local paths, for example:

```python
df_train = pd.read_csv('data/vad_dataset_train.csv')
```

**Dependencies:**

```bash
pip install numpy pandas matplotlib scikit-learn xgboost
```

---

## Part 1 – Quantization: uniform vs µ-law

**Goal:** compare 8-bit uniform quantization with 8-bit µ-law companding (ITU-T G.711, North American standard, µ = 255) on two synthetic signals sampled at 8 kHz.

| Signal | Description |
|--------|-------------|
| Vowel | Loud 800 Hz sine, amplitude 0.8 |
| Consonant | Quiet Gaussian noise, amplitude about 0.04 |

**Method.** Each signal is quantized in two ways, and the quality is measured with the empirical SQNR (signal-to-quantization-noise ratio):

- **Uniform:** quantize directly on 8 bits.
- **µ-law:** compress, quantize on 8 bits, then expand.

The core of the µ-law compression:

```python
def compress_mu_law(x, mu=255):
    return np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
```

**Results**

| Signal | SQNR uniform | SQNR µ-law |
|--------|-------------:|-----------:|
| Vowel (loud) | **42.08 dB** | 31.83 dB |
| Consonant (quiet) | 18.59 dB | **31.30 dB** |

![Quantization comparison](assets/part1_signals.png)

**Takeaway.** Uniform quantization has a fixed noise level, so quiet sounds suffer most. µ-law uses small steps for small amplitudes, so its SQNR stays nearly constant (about 31 dB) across loud and quiet signals. It loses about 10 dB on the loud vowel but gains about 12.7 dB on the quiet consonant. This is why µ-law suits speech, where amplitudes vary widely.

---

## Part 2 – Voice Activity Detection with ML

**Goal:** classify each audio frame as speech (`1`) or silence (`0`).

**Features (4):** `energy`, `zcr` (zero-crossing rate), `spectral_flatness`, `spectral_centroid`.

**Pipeline**

1. Load the train and test sets.
2. Check the class distribution.
3. Standardize the features (`StandardScaler`, fitted on the training set only).
4. Train two models and compare them on the test set:
   - Logistic Regression
   - Random Forest (`random_state=42`)

**Class distribution (training set)**

| Class | Count | Share |
|-------|------:|------:|
| 1 (speech) | 5,418 | 54.18 % |
| 0 (silence) | 4,582 | 45.82 % |

The classes are reasonably balanced, so accuracy is a meaningful metric here.

**Results (3,000 test frames)**

| Model | Accuracy | Precision | Recall | F1 |
|-------|---------:|----------:|-------:|---:|
| Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 |
| Random Forest | 1.00 | 1.00 | 1.00 | 1.00 |

| Logistic Regression | Random Forest |
|:---:|:---:|
| ![LogReg confusion matrix](assets/part2_confusion_logreg.png) | ![RF confusion matrix](assets/part2_confusion_rf.png) |

**Takeaway.** Both models reach essentially perfect scores, which suggests the dataset is very cleanly separable (likely synthetic). Even a linear model is enough, so the Random Forest brings no visible gain on this data. Real recordings with background noise would be a harder test.

---

## Part 3 – Bandwidth savings

**Goal:** estimate how much bandwidth silence suppression saves on a 64 kbit/s voice channel (DS0).

The **voice activity factor** α is the fraction of frames predicted as speech by the Random Forest:

```
α = mean(predictions)
effective rate = α × 64 kbit/s
savings = (1 − α) × 100 %
```

**Results**

| Metric | Value |
|--------|------:|
| Frames analyzed | 3,000 |
| Voice activity factor α | 0.560 |
| Effective average rate per channel | 35.86 kbit/s |
| **Bandwidth savings** | **43.97 %** |

**Takeaway.** Since the speaker is silent about 44 % of the time, not transmitting silence cuts the average bit rate nearly in half.

---

## Part 4 – Erlang B dimensioning with traffic forecasting

**Goal:** size a trunk group for a 1 % blocking probability, then check whether forecasting the traffic and adjusting capacity dynamically uses fewer resources than a static design.

### 4.1 Static dimensioning

The Erlang B blocking probability is computed with the standard recursion, and the smallest number of channels that meets the target is searched:

```python
def erlang_b(A, S):
    pb = 1.0
    for k in range(1, S + 1):
        pb = (A * pb) / (k + A * pb)
    return pb
```

| Parameter | Value |
|-----------|------:|
| Peak traffic | 35 Erlangs |
| Target blocking probability | 1 % |
| **Channels required** | **47** |
| T1 links (24 channels each) | 2 |

### 4.2 Dynamic dimensioning with XGBoost

An `XGBRegressor` predicts the offered traffic, and the capacity S(t) is recomputed at each time step from the prediction.

- **Features:** `hour`, `minute`, `is_peak_hour`, `traffic_lag_1` to `traffic_lag_4`
- **Target:** `offered_traffic_erlangs`
- **Split:** chronological, 70 % train / 30 % test (no shuffling, to respect time order)
- **Time step:** 15 minutes (0.25 h)

**Results (test window, 16:45 to 23:45)**

| Metric | Value |
|--------|------:|
| Dynamic resources | 229.2 channel-hours |
| Static resources (47 channels) | 340.8 channel-hours |
| **Gain** | **32.72 %** |

![Traffic forecast and dynamic capacity](assets/part4_traffic_forecast.png)

**Takeaway.** The dynamic capacity (green) follows the traffic and stays well below the static capacity (red) most of the time, which frees resources during off-peak hours.

---

## Summary of results

| Part | Key result |
|------|-----------|
| 1 | µ-law keeps SQNR near 31 dB for both signals, while uniform ranges from 18.6 to 42.1 dB |
| 2 | Logistic Regression and Random Forest both reach 100 % accuracy on the test set |
| 3 | α = 0.560, giving 43.97 % bandwidth savings |
| 4 | 47 channels (2 × T1) for static design; 32.72 % resource gain with dynamic scaling |

---

## Limitations and observations

- **Perfect VAD scores.** Scores of 1.00 for both models point to an easy, probably synthetic dataset. They should not be expected to carry over to noisy real-world audio.
- **Partial-day gain in Part 4.** The 32.72 % gain is computed on the 30 % test split only (16:45 to 23:45), which is the evening traffic decline. It is not a full 24 h figure, and a full-day average would likely be lower since peak hours need more capacity.
- **Forecast bias.** On the test window, the XGBoost predictions sit above the real traffic. This is a safe direction for dimensioning (capacity is over-provisioned rather than under-provisioned), but it means part of the possible savings is not captured.
- **No safety margin.** The dynamic capacity is computed from predicted traffic alone. A forecast that underestimates would cause blocking above 1 %. A margin or a prediction interval would make the approach safer.
- **Simple signals in Part 1.** The vowel and consonant are synthetic (a sine and white noise), which is a simplification of real speech.

---

## Tech stack

- **Python 3**, **Google Colab**
- **NumPy**, **pandas**, **Matplotlib**
- **scikit-learn** (`StandardScaler`, `LogisticRegression`, `RandomForestClassifier`, metrics)
- **XGBoost** (`XGBRegressor`)

---

## Course context

Course **MGR840**, École de technologie supérieure (ÉTS), Montréal. This repository is shared for portfolio purposes.
