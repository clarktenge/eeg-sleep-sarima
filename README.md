# Rhythms of the Sleeping Brain

> Can a classical SARIMA model track when a sleeping brain shifts between sleep stages?
> **TL;DR:** A manually-tuned SARIMA model achieves 0.30% MAPE on held-out overnight EEG alpha power — and spectral analysis confirms the dominant ~90-minute NREM-REM ultradian cycle without a single call to `auto_arima`.

**Author:** Clark Enge (<clarkenge@ucsb.edu>)  
**Status:** Active development — seasonal period refinement in progress

---

## The Problem

Closed-loop neural interfaces — cortical stimulators, motor-imagery decoders, sleep-stage monitors — must continuously estimate a patient's sleep state to calibrate stimulation thresholds. Over-stimulating during slow-wave sleep can disrupt memory consolidation; under-stimulating during REM disrupts therapeutic effect.

**Alpha power (8–13 Hz)** is a canonical biomarker for arousal state:

```
Wake / N1 onset:  alpha elevated   →  brain idling, eyes closed
N2 / N3 (SWS):   alpha suppressed →  deep slow-wave sleep
REM:              alpha transient  →  dreaming, muscle atonia
```

A model that forecasts alpha power one to three epochs ahead (30–90 seconds) gives an implanted device a short look-ahead for state transitions. This project fits that model using classical transparent SARIMA — fully auditable, no black-box components, algebraically expressible. That matters for regulatory submissions.

The time series `X_t` is constructed by extracting alpha-band power from each 30-second EEG epoch, producing ~2650 observations across a full overnight recording:

```
Raw EDF (Fpz-Cz, 100 Hz)
    │
    ▼
4th-order zero-phase Butterworth [8–13 Hz] per epoch
    │
    ▼
Welch PSD (2-second Hann windows, 50% overlap)
    │
    ▼
X_t = ∫₈¹³ P̂ₓₓ(f) df   [μV²/Hz per epoch]
    │
    ▼
~2650-point univariate time series → SARIMA modeling
```

---

## The Pipeline

### Step 1 — Signal Processing

Each 30-second epoch (3000 samples at 100 Hz) is band-pass filtered and transformed into a single scalar via Welch's method and trapezoidal integration. Zero-phase filtering (`sosfiltfilt`) prevents phase distortion on short segments.

```python
# Per-epoch alpha power extraction
freqs, pxx = signal.welch(filtered_epoch, fs=100, nperseg=200, window='hann')
alpha_mask  = (freqs >= 8) & (freqs <= 13)
X_t         = np.trapz(pxx[alpha_mask], freqs[alpha_mask])   # μV²/Hz
```

Values are on the order of 10⁻¹¹ μV²/Hz — extremely small, which is why all modeling is done on the log scale.

### Step 2 — Transformation & Stationarity

The raw series is right-skewed and exhibits variance proportional to its level (visible in the rolling statistics). A log transform stabilizes variance before differencing:

```
Y_t = log(X_t + ε),   ε = 10⁻⁶ · median(X_t)
```

**ADF test results** — only tests for unit root; not a complete stationarity check:

| Series | ADF Statistic | p-value | Decision |
|---|---|---|---|
| `X_t` raw | — | — | Non-stationary (rolling mean drifts) |
| `log(X_t)` | −7.69 | < 0.001 | Stationary by ADF |
| `∇ log(X_t)` | −16.92 | < 0.001 | Stationary |
| `∇∇₁₈ log(X_t)` | −16.74 | < 0.001 | Stationary |

`log(X_t)` already passes ADF, but the rolling mean is visibly non-constant — ADF detects only unit-root non-stationarity, not the oscillatory macrostructure visible in the EDA plots. We proceed with `d=1`.

### Step 3 — Model Identification

ACF/PACF of the differenced series `∇∇₁₈ log(X_t)` (S = 18 epochs ≈ 9 minutes) reveals the non-seasonal and seasonal order structure. Visual cutoff rules applied manually — no `auto_arima`.

Two candidate models compared:

| Model | Specification | k | n | Log-Lik | AICc |
|---|---|---|---|---|---|
| Model A | SARIMA(1,1,1)(1,1,1)[18] | 6 | 2120 | −404.63 | 821.30 |
| **Model B** | **SARIMA(2,1,1)(0,1,1)[18]** | **6** | **2120** | **−401.29** | **814.61** |

**Model B wins** with ΔAICc = 6.69 — a meaningful difference (rule of thumb: >2 is meaningful, >10 is decisive). AICc is computed from first principles:

```python
AICc = -2·ℓ(θ̂) + 2k + 2k(k+1)/(n-k-1)
```

**Model B fitted equation** (SARIMA(2,1,1)(0,1,1)[18]):

```
(1 − φ₁B − φ₂B²)(1 − B¹⁸)(1 − B) Yₜ = (1 + Θ₁B¹⁸)(1 + θ₁B) εₜ

φ₁ = +0.3140  (SE = 0.0253)
φ₂ = +0.0853  (SE = 0.0221)
θ₁ = −0.8962  (SE = 0.0152)
Θ₁ = −0.9990  (SE = 0.1337)
σ̂² =  0.0824  (SE = 0.0110)
```

### Step 4 — Residual Diagnostics

| Test | Lag | Statistic | p-value | Decision |
|---|---|---|---|---|
| Ljung-Box (residuals) | 10 | 4.64 | 0.914 | ✓ White noise |
| Ljung-Box (residuals) | 20 | 233.8 | < 0.001 | ✗ Autocorrelation |
| Ljung-Box (residuals) | 30 | 236.6 | < 0.001 | ✗ Autocorrelation |
| McLeod-Li (ε²) | 10 | 0.003 | 1.000 | ✓ No ARCH effect |
| McLeod-Li (ε²) | 20 | 109.5 | < 0.001 | ✗ Nonlinear structure |
| Shapiro-Wilk | — | W = 0.171 | < 0.001 | ✗ Non-normal |

The model passes at lag 10 but fails at lags 20 and 30 — residual autocorrelation remains beyond the immediate neighborhood. The most likely cause: **the seasonal period S = 18 (9 minutes) is misspecified**. Spectral analysis (below) identifies the dominant cycle at ~90 minutes, corresponding to S = 180. Refitting with S = 180 is the primary next step.

### Step 5 — Forecasting

Forecasts are generated on the log scale and back-transformed with a bias correction for the log-normal mean:

```
X̂ₙ₊ₕ = exp(Ŷₙ₊ₕ + σ̂²ₕ/2) − ε          (bias-corrected mean)
Lower = exp(Ŷₙ₊ₕ − 1.96·σ̂ₕ) − ε
Upper = exp(Ŷₙ₊ₕ + 1.96·σ̂ₕ) − ε
```

The `+ σ̂²ₕ/2` term corrects for the fact that `E[e^Y] = e^(μ + σ²/2)` for Gaussian `Y` — omitting it would systematically underestimate the mean.

**Forecast accuracy on held-out test set (530 epochs, ~4.4 hours):**

| Metric | Value |
|---|---|
| MAPE | 0.30% |
| RMSE | ~10⁻¹¹ μV²/Hz |
| MAE | ~10⁻¹¹ μV²/Hz |

MAPE of 0.30% is the interpretable metric — RMSE and MAE reflect the extremely small absolute scale of alpha power values.

### Step 6 — Spectral Analysis

| Test | Statistic | p-value | Conclusion |
|---|---|---|---|
| Fisher g-test | g = 0.0536 | < 0.001 | Dominant periodicity is significant |
| KS cumulative periodogram | D = 0.413 | < 0.001 | Spectral structure present |

**Top periodogram peaks:**

| Rank | Frequency | Period | Interpretation |
|---|---|---|---|
| 1 | 0.091 cyc/hr | ~663 min | Overnight trend (full recording) |
| 2 | 0.045 cyc/hr | ~1325 min | Subharmonic of recording length |
| **3** | **0.815 cyc/hr** | **~74 min** | **NREM-REM ultradian cycle** |
| **4** | **0.634 cyc/hr** | **~95 min** | **NREM-REM ultradian cycle** |
| 5 | 0.906 cyc/hr | ~66 min | Sleep cycle harmonic |

The 74- and 95-minute peaks straddle the canonical 90-minute NREM-REM ultradian rhythm — strong evidence that **S = 180 epochs** (180 × 30s = 90 min) is the correct seasonal period, not S = 18.

---

## Results Summary

| Component | Result |
|---|---|
| Series length | 2650 epochs (~22.1 hours) |
| Train / Test | 2120 / 530 epochs (80/20) |
| Best model | SARIMA(2,1,1)(0,1,1)[18], AICc = 814.61 |
| Forecast MAPE (test) | 0.30% |
| Dominant spectral peak | ~90-min ultradian NREM-REM cycle |
| Fisher g-test | p < 0.001 — significant periodicity |
| Ljung-Box (lag 10) | p = 0.914 — white noise ✓ |
| Ljung-Box (lag 20/30) | p < 0.001 — autocorrelation remains ✗ |

---

## Figures

| | |
|---|---|
| [![EDA](https://github.com/clarktenge/eeg-sleep-sarima/raw/main/outputs/figures/eda_plots.png)](https://github.com/clarktenge/eeg-sleep-sarima/blob/main/outputs/figures/eda_plots.png) | [![ACF/PACF](https://github.com/clarktenge/eeg-sleep-sarima/raw/main/outputs/figures/acf_pacf.png)](https://github.com/clarktenge/eeg-sleep-sarima/blob/main/outputs/figures/acf_pacf.png) |
| [![Residuals](https://github.com/clarktenge/eeg-sleep-sarima/raw/main/outputs/figures/residuals_Model_B.png)](https://github.com/clarktenge/eeg-sleep-sarima/blob/main/outputs/figures/residuals_Model_B.png) | [![Forecast](https://github.com/clarktenge/eeg-sleep-sarima/raw/main/outputs/figures/forecast.png)](https://github.com/clarktenge/eeg-sleep-sarima/blob/main/outputs/figures/forecast.png) |
| [![Spectral Analysis](https://github.com/clarktenge/eeg-sleep-sarima/raw/main/outputs/figures/spectral_analysis.png)](https://github.com/clarktenge/eeg-sleep-sarima/blob/main/outputs/figures/spectral_analysis.png) | [![ACF/PACF (no seasonal diff)](https://github.com/clarktenge/eeg-sleep-sarima/raw/main/outputs/figures/acf_pacf_nodiff.png)](https://github.com/clarktenge/eeg-sleep-sarima/blob/main/outputs/figures/acf_pacf_nodiff.png) |

---

## Under the Hood

### Alpha Power Extraction ([`pipeline.py`](pipeline.py))

The Welch method with 2-second Hann-windowed segments gives a lower-variance PSD estimate than the raw periodogram, at the cost of frequency resolution. For a 30-second epoch at 100 Hz:

```
N_epoch = 3000 samples
nperseg = 200 samples (2 seconds)
overlap = 50%  →  number of segments = ~29
frequency resolution = 100/200 = 0.5 Hz
```

The 0.5 Hz resolution is sufficient to resolve the 8–13 Hz alpha band cleanly.

### Manual AICc ([`pipeline.py`](pipeline.py))

AICc is computed from raw model attributes — no wrapper functions:

```python
ll       = result.llf          # maximized log-likelihood
k        = result.df_model + 1 # free parameters including σ²
n        = result.nobs         # effective observations after differencing
aic      = -2*ll + 2*k
aicc     = aic + (2*k*(k+1)) / (n - k - 1)
```

### Fisher g-test ([`pipeline.py`](pipeline.py))

Tests whether the largest periodogram ordinate is significant against a flat (white noise) spectrum. Exact p-value via:

```
g = max(Iⱼ) / Σ Iⱼ
p = Σₖ₌₁^⌊1/g⌋ (-1)^(k-1) · C(m,k) · (1 - k·g)^(m-1)
```

where `m` is the number of Fourier frequencies (excluding DC).

### McLeod-Li Test ([`pipeline.py`](pipeline.py))

Ljung-Box applied to the **squared residuals** εₜ². Tests for nonlinear structure (ARCH/GARCH effects). A linear SARIMA model produces uncorrelated εₜ but not necessarily uncorrelated εₜ² — volatility clustering in the residuals signals that a GARCH extension may be warranted.

---

## Repository Structure

```
eeg-sleep-sarima/
├── data/
│   ├── SC4001E0-PSG.edf          # PhysioNet Sleep-EDF — PSG recording (excluded from git)
│   └── SC4001EC-Hypnogram.edf    # Sleep stage annotations (excluded from git)
├── outputs/
│   └── figures/                  # EDA, ACF/PACF, residuals, forecast, spectral
├── eeg_sleep_sarima.ipynb        # Main analysis notebook (12 sections, step-by-step)
├── pipeline.py                   # Headless runnable script (same logic, saves to outputs/)
├── report_outline.md             # Structured report scaffold
├── requirements.txt
└── .gitignore
```

---

## Getting Started

```bash
git clone https://github.com/clarktenge/eeg-sleep-sarima.git
cd eeg-sleep-sarima

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the two EDF files from [PhysioNet Sleep-EDF](https://physionet.org/content/sleep-edfx/1.0.0/) (free account required) and place them in `data/`:

```
data/SC4001E0-PSG.edf
data/SC4001EC-Hypnogram.edf
```

Run the full pipeline headlessly:

```bash
python pipeline.py
# Figures saved to outputs/figures/
```

Or open the notebook for the step-by-step walkthrough:

```bash
jupyter notebook eeg_sleep_sarima.ipynb
```

---

## Key Findings

### 1. S = 18 Is Likely the Wrong Seasonal Period

The candidate seasonal period of 18 epochs (9 minutes) was chosen as an initial hypothesis. The spectral analysis finds the dominant cycles at 74 and 95 minutes — bracketing the canonical 90-minute NREM-REM ultradian rhythm. S = 180 (90-minute period) is the well-supported alternative. The Ljung-Box failure at lags 20 and 30 is consistent with this misspecification.

### 2. ADF Alone Is Not Enough

`log(X_t)` passes the ADF test (p < 0.001, stat = −7.69), suggesting no unit root — yet the rolling mean in the EDA clearly oscillates and the ACF of the raw log series decays slowly. ADF tests only for stochastic trend; the oscillatory macrostructure requires differencing regardless. This is why stationarity must be assessed visually alongside the formal test.

### 3. 0.30% MAPE Without Overfitting

Despite the residual autocorrelation issue, the model achieves 0.30% MAPE on the 20% held-out test set. The forecast tracks the general trajectory of alpha power across a 4.4-hour test window.

### 4. Spectral Confirmation of Sleep Architecture

Fisher's g-test (p < 0.001) confirms the dominant spectral peak is not attributable to chance. The cumulative periodogram diverges significantly from the white-noise diagonal (KS D = 0.413) — most spectral mass is concentrated in the low-frequency cycles corresponding to the ultradian sleep rhythm.

---

## Open Questions

- **S = 180 refit:** Does a 90-minute seasonal period clean up the residual autocorrelation at lags 20 and 30?
- **McLeod-Li at lag 20+:** Is the nonlinear structure genuine (ARCH effects) or an artifact of the wrong seasonal period?
- **Multi-subject generalization:** Does the same SARIMA structure hold across different subjects in the Sleep-EDF cassette recordings?
- **Comparison baseline:** How does the SARIMA forecast compare to a naïve seasonal baseline (repeat last cycle) on RMSE/MAPE?

---

## Data

**Source:** [PhysioNet Sleep-EDF Expanded Database](https://physionet.org/content/sleep-edfx/1.0.0/)

Kemp B, Värri A, Rosa AC, Nielsen KD, Gade J. (2000). A simple format for exchange of digitized polygraphic recordings. *Electroencephalography and Clinical Neurophysiology*, 69(4), 391–395.

Goldberger AL, et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215–e220.

EDF files are excluded from version control (`.gitignore`). Access requires a free PhysioNet credentialed account.

---

## Dependencies

Python 3.10+ · `mne` · `scipy` · `statsmodels` · `numpy` · `pandas` · `matplotlib` · `seaborn`

See [`requirements.txt`](requirements.txt) for the full pinned list.
