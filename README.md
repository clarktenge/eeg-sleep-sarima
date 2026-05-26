# Rhythms of the Sleeping Brain
### SARIMA Modeling and Spectral Analysis of EEG Alpha Power Across Human Sleep Cycles

> **Status:** Active development — preprocessing pipeline complete, SARIMA model tuning in progress.

---

## Overview

This project applies classical statistical time series methods to overnight EEG recordings, constructing a univariate alpha-band power series (8–13 Hz) from 30-second polysomnography epochs and fitting a manually-tuned SARIMA model to it.

The motivating engineering problem is BCI state calibration. Closed-loop neural interfaces — cortical stimulators, motor-imagery decoders, sleep-monitoring implants — must track a user's sleep/wake state in real time to adjust signal thresholds and avoid over-stimulation during slow-wave sleep. Alpha power is a reliable proxy for arousal state: it suppresses during deep sleep (N2/N3), rises at sleep onset (N1), and is transiently elevated during REM. A model that forecasts its near-term trajectory gives a device a one- to three-epoch look-ahead for state switching.

The project deliberately avoids automated modeling tools (`auto_arima`, `check_residuals`) to demonstrate transparent, auditable methodology — a requirement for any analysis pipeline targeting regulated medical-device contexts.

**Stack:** `mne` · `scipy` · `statsmodels` · `numpy` · `pandas` · `matplotlib` · `seaborn`

---

## Repository Structure

```
eeg-sleep-sarima/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── pipeline.py               # Modular signal processing + SARIMA pipeline
│
├── data/
│   └── .gitkeep              # EDF files excluded via .gitignore (see Data section)
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory analysis — raw series, rolling stats
│   ├── 02_stationarity.ipynb # ADF testing, transformation selection
│   └── 03_model_selection.ipynb  # ACF/PACF inspection, AICc comparison
│
├── outputs/
│   ├── figures/              # EDA plots, ACF/PACF, residual diagnostics, forecast overlay
│   └── models/               # Serialized SARIMAX result objects (.pkl)
│
└── report/
    └── report_outline.md     # Structured section-by-section report scaffold
```

---

## Data

**Source:** [PhysioNet Sleep-EDF Expanded Database](https://physionet.org/content/sleep-edfx/1.0.0/)
Kemp B, et al. (2000). *Electroenceph. Clin. Neurophysiol.* 69(4), 391–395.
Goldberger AL, et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23), e215–e220.

**Recording used:** `SC4001E0-PSG.edf` — Subject 1, whole-night cassette PSG, channel `EEG Fpz-Cz`.
**Hypnogram:** `SC4001EC-Hypnogram.edf` — 30-second staging annotations (W, N1, N2, N3, REM).

Place EDF files in `data/` before running the pipeline. They are excluded from version control (see `.gitignore`).

---

## Data Pipeline Architecture

```
Raw EDF (Fpz-Cz, 100 Hz)
        │
        ▼
1. LOADING
   mne.io.read_raw_edf → resample to 100 Hz → extract single-channel array

        │
        ▼
2. BAND-PASS FILTERING  (per 30-second epoch)
   4th-order zero-phase Butterworth, passband [8–13 Hz]
   sosfiltfilt → avoids phase distortion on short segments

        │
        ▼
3. POWER EXTRACTION  (per epoch)
   Welch's method: 2-second Hann windows, 50% overlap
   Integrate PSD over [8–13 Hz] → scalar alpha power X_t
   Output: univariate series of ~900–1020 30-second epochs

        │
        ▼
4. TRAIN / TEST SPLIT
   80% training (~720 epochs) | 20% held-out test (~180 epochs)

        │
        ▼
5. EDA
   Raw series plot · rolling mean/std · marginal histogram
   Visual assessment: trend, variance structure, outliers

        │
        ▼
6. TRANSFORMATION & STATIONARITY
   Log-transform Y_t = log(X_t + ε)  →  stabilise multiplicative variance
   Regular differencing  ∇Y_t = Y_t − Y_{t−1}
   Seasonal differencing ∇_S Y_t  (S = 18 or 180 epochs)
   ADF test at each stage → confirm stationarity before identification

        │
        ▼
7. MODEL IDENTIFICATION
   ACF / PACF of differenced series → read non-seasonal (p, q) and
   seasonal (P, Q) orders by visual cutoff / tail-off pattern

        │
        ▼
8. MANUAL SARIMA FITTING & SELECTION
   statsmodels SARIMAX — at least two candidate specifications
   AICc computed manually:  AICc = AIC + 2k(k+1) / (n − k − 1)
   No auto-selection wrappers

        │
        ▼
9. RESIDUAL DIAGNOSTICS
   Residual time plot · ACF of residuals · Q-Q plot
   Ljung-Box (lags 10, 20, 30) · Shapiro-Wilk
   All from raw residual arrays — no check_residuals()

        │
        ▼
10. FORECASTING & BACK-TRANSFORMATION
    Out-of-sample prediction on log scale
    Bias-corrected inversion: X̂ = exp(Ŷ + σ²_h/2) − ε
    95% CI: [exp(Ŷ − 1.96σ_h), exp(Ŷ + 1.96σ_h)]
    Overlay forecast vs. actual on held-out test window

        │
        ▼
11. SPECTRAL ANALYSIS
    Periodogram (Schuster) — identify dominant cycles in cycles/hour
    Parametric AR PSD (Yule-Walker, order by AIC) — sharpen peak localisation
    Expected: ~0.67 cycles/hour (~90-min ultradian NREM-REM cycle)
```

---

## Installation & Setup

Requires Python ≥ 3.10.

```bash
git clone https://github.com/<your-handle>/eeg-sleep-sarima.git
cd eeg-sleep-sarima
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the PhysioNet EDF files and place them in `data/`:

```bash
# Using wget (Linux/macOS) or browser-download the files manually
wget -P data/ https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001E0-PSG.edf
wget -P data/ https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4001EC-Hypnogram.edf
```

Run the full pipeline:

```bash
python pipeline.py --edf data/SC4001E0-PSG.edf --hypno data/SC4001EC-Hypnogram.edf
```

---

## Current Status & Roadmap

### Completed
- [x] Repository structure and module architecture defined
- [x] EDF loading and channel extraction via MNE (`load_edf_signal`)
- [x] Alpha-band power extraction via Welch's method (`extract_alpha_power`)
- [x] Train/test split and EDA plotting scaffold
- [x] ADF stationarity testing pipeline with manual output (`check_stationarity`)
- [x] Log-transform and manual differencing functions
- [x] ACF/PACF visualization with confidence intervals (`plot_acf_pacf`)
- [x] Manual AICc formula implemented (`compute_manual_aicc`)
- [x] Residual diagnostic suite (Ljung-Box, Shapiro-Wilk, Q-Q)
- [x] Spectral analysis module scaffold (`compute_spectral_density`)
- [x] Report outline with section-level analytical guidance

### In Progress
- [ ] Manual SARIMA order identification from ACF/PACF plots (visual inspection)
- [ ] Fitting and comparing two SARIMA candidate models on training data
- [ ] Residual validation against white-noise assumptions
- [ ] Back-transformation and 95% CI construction on test window
- [ ] Spectral peak verification at ~90-minute ultradian frequency

### Planned
- [ ] Multi-subject generalization across Sleep-EDF cassette recordings
- [ ] Comparison of SARIMA forecast vs. naive seasonal baseline (RMSE/MAPE)
- [ ] Final written report with algebraic model specification

---

## References

1. Kemp B, Värri A, Rosa AC, Nielsen KD, Gade J. (2000). A simple format for exchange of digitized polygraphic recordings. *Electroencephalography and Clinical Neurophysiology*, 69(4), 391–395.
2. Goldberger AL, et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215–e220.
3. Box GEP, Jenkins GM, Reinsel GC, Ljung GM. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
4. Hurvich CM, Tsai CL. (1989). Regression and time series model selection in small samples. *Biometrika*, 76(2), 297–307.
5. Rechtschaffen A, Kales A. (1968). *A Manual of Standardized Terminology, Techniques and Scoring System for Sleep Stages of Human Subjects*. US Government Printing Office.

---

*Dataset access requires a PhysioNet credentialed account. See [physionet.org/content/sleep-edfx](https://physionet.org/content/sleep-edfx/1.0.0/) for terms.*
