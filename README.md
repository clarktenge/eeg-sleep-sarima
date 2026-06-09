# Rhythms of the Sleeping Brain

> Can a classical, fully hand-built time-series model track overnight EEG alpha power - and does the "obvious" 90-minute sleep cycle actually belong in the model?
> 
> **TL;DR:** I model overnight EEG alpha power with an ARIMA fit entirely by hand (no `auto_arima`). The physiologically obvious seasonal period (S = 180 ≈ 90 min) turns out to be wrong. ADF rejects a unit root in log(X_t), and fitting a d=1 family reveals a near-unit MA root (1.006) — the signature of over-differencing — so the final model is a non-seasonal **ARMA(2,0,1)** on log(X_t) (AICc = 698.33), with the ultradian rhythm documented through spectral analysis instead.

**Author:** Clark Enge (<clarkenge@ucsb.edu>)
**Course:** PSTAT W 274 — Time Series Analysis

---

## The Problem

Closed-loop neural interfaces — cortical stimulators, sleep-stage monitors — need a running estimate of a patient's sleep state to calibrate stimulation. **Alpha power (8–13 Hz)** is a convenient proxy for arousal state:

```
Wake / N1 onset:  alpha elevated   ->  brain idling, eyes closed
N2 / N3 (SWS):    alpha suppressed ->  deep slow-wave sleep
REM:              alpha transient  ->  dreaming, muscle atonia
```

A model that forecasts alpha power a step or two ahead gives such a device a short look-ahead. I build that model with classical, transparent ARIMA — every step done by hand so the result is fully auditable.

The series `X_t` is built by extracting alpha-band power from each 30-second EEG epoch, giving ~2650 observations across the overnight recording:

```
Raw EDF (Fpz-Cz, 100 Hz)
    -> 4th-order zero-phase Butterworth [8-13 Hz] per epoch
    -> Welch PSD (2 s Hann windows, 50% overlap)
    -> X_t = integral of PSD over 8-13 Hz   [uV^2/Hz per epoch]
    -> ~2650-point univariate series -> ARMA modeling
```

---

## The Pipeline

### Step 1 — Signal processing

Each 30-second epoch (3000 samples at 100 Hz) is band-pass filtered and reduced to one scalar via Welch's method and trapezoidal integration. Zero-phase filtering (`sosfiltfilt`) avoids the phase lag a causal filter would add on such short segments. Values land around 10^-11 uV2/Hz, which is why all modeling happens on the log scale.

### Step 2 — Transformation & stationarity

The rolling standard deviation tracks the rolling mean (variance proportional to level), so I log-transform before modeling:

```
Y_t = log(X_t + eps),   eps = 1e-6 * median(X_t)
```

ADF results (the test only detects a unit root — not seasonality, variance changes, or breaks):

| Series | ADF stat | p-value | Decision |
|---|---|---|---|
| log(X_t) | -7.69 | ~1e-11 | Stationary by ADF |
| d1 log(X_t) | -16.92 | ~1e-29 | Stationary |

log(X_t) strongly rejects the unit-root null. ADF alone is not decisive (it only tests the AR part), so I also fit the d=1 candidate family as a check — and that check confirms d=0: the best d=1 model, ARIMA(2,1,2), returns an MA root of 1.006, the textbook signature of over-differencing. The d=0 candidate ARMA(2,0,1) has a well-separated MA root (1.327) and edges out ARIMA(2,1,2) on AICc. Both criteria agree: **d=0, no differencing needed.**

### Step 3 — Testing (and rejecting) the seasonal heuristic

The NREM-REM cycle is ~90 min = 180 epochs, so a seasonal SARIMA at S = 180 looks obvious. I tested it and rejected it on three independent grounds:

| Check | Result |
|---|---|
| Over-differencing (variance should drop) | var rises 0.111 to 0.230 (S=18) / 0.220 (S=180) — over-differencing |
| Seasonal ACF/PACF spike at lag 18/147/180/189 | none significant (all inside +-0.043 band) |
| Fit a state-space SARIMA at S=180 | ~180-dim state, does not converge in reasonable time |

The ultradian rhythm is real but **spectrally broad** (66-95 min; see Step 7), so no single seasonal lag captures it. I model the mean with a non-seasonal ARMA and treat the rhythm as a spectral finding.

### Step 4 — Model identification & selection

With d=0 settled, I read the ACF and PACF of Y_t = log(X_t) directly. The ACF decays slowly across many lags; the PACF has significant spikes at lags 1 and 2 before falling inside the 95% band — the canonical **AR(2)** pattern. I then compared d=1 and d=0 candidate families by hand-coded AICc:

**d=1 candidates** (fitted for the over-differencing diagnostic; AICc not comparable to d=0):

| Model | AICc | Min MA root |
|---|---|---|
| ARIMA(0,1,1) | 791.4 | 1.505 |
| ARIMA(1,1,1) | 730.6 | 1.176 |
| ARIMA(2,1,1) | 723.8 | 1.115 |
| ARIMA(1,1,2) | 717.2 | 1.052 |
| ARIMA(2,1,2) | 698.9 | **1.006** (over-differencing) |

**d=0 candidates** (AICc comparable within this group):

| Model | AICc |
|---|---|
| ARMA(1,0,1) | 722.4 |
| ARMA(1,0,2) | 702.0 |
| ARMA(2,0,2) | 725.3 |
| **ARMA(2,0,1)** | **698.33** (winner) |

**ARMA(2,0,1) wins** (delta AICc ~3.7 over the next-best d=0 candidate; root diagnostics all well-separated).

**Fitted model** (log scale, Y_t = log(X_t + eps)):

```
(1 - 1.157 B + 0.197 B^2)(Y_t + 25.75) = (1 - 0.753 B) eps_t

phi_1 = +1.1575 (SE 0.037)    theta_1 = -0.7535 (SE 0.029)
phi_2 = -0.1973 (SE 0.032)    sigma^2 = +0.0810 (SE 0.002)
mu    = -25.75  (long-run mean of log series)
```

### Step 5 — Residual diagnostics

| Test | Result | Verdict |
|---|---|---|
| Ljung-Box (residuals, lags 10/20/30) | p = 0.71 / 0.82 / 0.69 | white noise |
| Shapiro-Wilk (first 500) | W = 0.989, p ~ 9e-4 | mild non-normality (sensitive at n~2118) |
| McLeod-Li (Ljung-Box on eps^2) | p ~ 1e-8 | **ARCH effect** |

The mean model is clean (Ljung-Box passes), but McLeod-Li shows volatility clustering in the squared residuals — a genuine conditional-heteroscedasticity effect, and the clearest motivation for a future ARMA+GARCH extension.

### Step 6 — Forecasting

I report the **1-step-ahead** forecast (fixed parameters, filtered through the test set) — the honest metric for a short-horizon model, since a 530-step static forecast just reverts to the mean. Back-transform uses the log-normal bias correction X_hat = exp(Y_hat + sigma^2/2) - eps.

| Forecaster | Test MAPE |
|---|---|
| **ARMA(2,0,1) 1-step** | **19.6%** |
| Persistence (lag 1) | 23.2% |
| Naive seasonal (S = 180) | 31.2% |

The model beats both baselines. MASE (robust to near-zero values in deep sleep) is 0.860, confirming genuine improvement over the naive.

### Step 7 — Spectral analysis

Periodogram to locate seasonality (on the linearly detrended log series); Fisher's g-test and the KS cumulative-periodogram test to check the **residuals** are white (both implemented from scratch — R equivalents are GeneCycle::fisher.g.test and stats::cpgram). The theoretical spectral density of the fitted ARMA(2,0,1) is also plotted against the periodogram.

| Test | Statistic | p-value | Conclusion |
|---|---|---|---|
| Fisher g (series) | g = 0.081 | ~5.6e-46 | strong periodicity in X_t |
| Fisher g (residuals) | g = 0.008 | 0.198 | no leftover periodicity |
| KS cumulative periodogram (residuals) | D = 0.019 | 0.856 | residuals ~ white noise |

Dominant periodogram peaks: **73.6, 94.6, 66.2, 69.7 min** — a band around the ~90-minute NREM-REM cycle. The spread across periods is exactly why a single seasonal lag failed in Step 3. The ARMA(2,0,1) theoretical spectral density shows a smooth 1/f-type decay consistent with the near-unit AR root (1.053), capturing the short-range dependence without encoding the ultradian rhythm as a line frequency.

---

## Results Summary

| Component | Result |
|---|---|
| Series length | 2650 epochs (~22.1 hours) |
| Train / Test | 2120 / 530 epochs (80/20) |
| Best model | ARMA(2,0,1) on log(X_t), AICc = 698.33 |
| Ljung-Box (residuals) | p = 0.71 / 0.82 / 0.69 — white noise |
| McLeod-Li | p ~ 1e-8 — ARCH effect (future GARCH work) |
| Forecast MAPE (1-step) | 19.6% (beats persistence 23.2%, naive 31.2%) |
| Dominant spectral band | ~66-95 min ultradian NREM-REM cycle |
| Residual Fisher / KS | p = 0.198 / 0.856 — residuals spectrally white |

---

## Figures

All figures are regenerated by `pipeline.py` into `outputs/figures/`:

`eda_plots.png` · `acf_pacf.png` · `residuals_ARIMA212.png` · `forecast.png` · `spectral_analysis.png`

---

## Repository Structure

```
eeg-sleep-sarima/
├── data/                       # PhysioNet EDF files (excluded from git)
├── outputs/figures/            # EDA, ACF/PACF, residuals, forecast, spectral
├── eeg_sleep_sarima.ipynb      # Main analysis notebook (step-by-step, executed)
├── pipeline.py                 # Headless runnable script (same logic)
├── report_outline.md           # Report draft / outline
├── requirements.txt
└── .gitignore
```

---

## Getting Started

```bash
git clone https://github.com/clarktenge/eeg-sleep-sarima.git
cd eeg-sleep-sarima
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the EDF files from [PhysioNet Sleep-EDF](https://physionet.org/content/sleep-edfx/1.0.0/) into `data/`, then:

```bash
python pipeline.py                  # regenerates every figure in outputs/figures/
# or open eeg_sleep_sarima.ipynb for the step-by-step walkthrough
```

---

## Key Findings

1. **d=0, not d=1.** ADF rejects a unit root in log(X_t), and fitting the d=1 family confirms over-differencing via a near-unit MA root (1.006) in the best d=1 candidate. The final model operates on the log level directly.
2. **The 90-minute seasonal model is the wrong tool here.** It over-differences, has no ACF/PACF signature, and is computationally impractical. The honest move was to drop it and capture the rhythm spectrally.
3. **AICc and root diagnostics together select ARMA(2,0,1).** The ACF/PACF reading suggested AR(2); AICc confirmed the AR(2) skeleton and added one MA correction term that the residual diagnostics jointly justify.
4. **The residuals are linearly white but nonlinearly structured.** Ljung-Box and the spectral tests pass, but McLeod-Li reveals an ARCH effect — a clear, honest direction for future GARCH work.

---

## Data

[PhysioNet Sleep-EDF Expanded Database](https://physionet.org/content/sleep-edfx/1.0.0/). Kemp B. et al. (2000), Electroenceph. Clin. Neurophysiol. 69(4); Goldberger A.L. et al. (2000), Circulation 101(23). EDF files are excluded from version control; access requires a free PhysioNet account.

## Dependencies

Python 3.10+ · `mne` · `scipy` · `statsmodels` · `numpy` · `pandas` · `matplotlib` · `seaborn`. See `requirements.txt`.# Rhythms of the Sleeping Brain

> Can a classical, fully hand-built time-series model track overnight EEG alpha power - and does the "obvious" 90-minute sleep cycle actually belong in the model?
> 
> **TL;DR:** I model overnight EEG alpha power with an ARIMA fit entirely by hand (no `auto_arima`). The physiologically obvious seasonal period (S = 180 ≈ 90 min) turns out to be wrong. It over-differences the series and has no ACF/PACF signature, so the final model is a non-seasonal **ARIMA(2,1,2)**, with the ultradian rhythm documented through spectral analysis instead.

**Author:** Clark Enge (<clarkenge@ucsb.edu>)
**Course:** PSTAT W 274 — Time Series Analysis

---

## The Problem

Closed-loop neural interfaces — cortical stimulators, sleep-stage monitors — need a running estimate of a patient's sleep state to calibrate stimulation. **Alpha power (8–13 Hz)** is a convenient proxy for arousal state:

```
Wake / N1 onset:  alpha elevated   ->  brain idling, eyes closed
N2 / N3 (SWS):    alpha suppressed ->  deep slow-wave sleep
REM:              alpha transient  ->  dreaming, muscle atonia
```

A model that forecasts alpha power a step or two ahead gives such a device a short look-ahead. I build that model with classical, transparent ARIMA — every step done by hand so the result is fully auditable.

The series `X_t` is built by extracting alpha-band power from each 30-second EEG epoch, giving ~2650 observations across the overnight recording:

```
Raw EDF (Fpz-Cz, 100 Hz)
    -> 4th-order zero-phase Butterworth [8-13 Hz] per epoch
    -> Welch PSD (2 s Hann windows, 50% overlap)
    -> X_t = integral of PSD over 8-13 Hz   [uV^2/Hz per epoch]
    -> ~2650-point univariate series -> ARIMA modeling
```

---

## The Pipeline

### Step 1 — Signal processing

Each 30-second epoch (3000 samples at 100 Hz) is band-pass filtered and reduced to one scalar via Welch's method and trapezoidal integration. Zero-phase filtering (`sosfiltfilt`) avoids the phase lag a causal filter would add on such short segments. Values land around 10⁻¹¹ μV²/Hz, which is why all modeling happens on the log scale.

### Step 2 — Transformation & stationarity

The rolling standard deviation tracks the rolling mean (variance proportional to level), so I log-transform before differencing:

```
Y_t = log(X_t + eps),   eps = 1e-6 * median(X_t)
```

ADF results (the test only detects a unit root — not seasonality, variance changes, or breaks):

| Series | ADF stat | p-value | Decision |
|---|---|---|---|
| `log(X_t)` | −7.69 | ~1e-11 | Stationary by ADF |
| `∇ log(X_t)` | −16.92 | ~1e-29 | Stationary |

`log(X_t)` already passes ADF, but the rolling mean wanders and the log level decays slowly, so I still take one regular difference (`d = 1`) for practical ARMA modeling. This is exactly the ADF caveat in action: passing the test is not the same as being stationary.

### Step 3 — Testing (and rejecting) the seasonal heuristic

The NREM-REM cycle is ~90 min = 180 epochs, so a seasonal SARIMA at S = 180 looks obvious. I tested it and rejected it on three independent grounds:

| Check | Result |
|---|---|
| Over-differencing (variance should *drop*) | var rises 0.111 → 0.230 (S=18) / 0.220 (S=180) → **over-differencing** |
| Seasonal ACF/PACF spike at lag 18/147/180/189 | none significant (all inside ±0.043 band) |
| Fit a state-space SARIMA at S=180 | ~180-dim state, does not converge in reasonable time |

The ultradian rhythm is real but **spectrally broad** (66–95 min; see Step 6), so no single seasonal lag captures it. I model the mean with a non-seasonal ARIMA and treat the rhythm as a spectral finding.

### Step 4 — Model identification & selection

ACF of `∇ log(X_t)` shows one big spike at lag 1 (−0.38) then nothing; PACF tails off → first guess **MA(1) = ARIMA(0,1,1)**. I then compared candidates by a hand-coded AICc:

| Model | AICc | Ljung-Box (10/20/30) |
|---|---|---|
| ARIMA(0,1,1) — ACF/PACF guess | 793.4 | fails (~1e-9) |
| ARIMA(1,1,1) | 732.6 | borderline |
| ARIMA(2,1,1) | 725.8 | passes |
| ARIMA(1,1,2) | 719.2 | passes |
| **ARIMA(2,1,2)** | **700.9** | **passes (0.71 / 0.82 / 0.69)** |

**ARIMA(2,1,2) wins** (ΔAICc ≈ 18 over the next-best — decisive). Note the AICc model is *not* the ACF/PACF guess: the simple MA(1) fails its residual tests, so diagnostics + AICc pushed me to the richer model.

**Fitted model** (log scale, `Y_t = log(X_t + eps)`):

```
(1 - 1.134 B + 0.187 B^2)(1 - B) Y_t = (1 - 1.730 B + 0.731 B^2) eps_t

phi_1 = +1.1340 (SE 0.044)    theta_1 = -1.7296 (SE 0.036)
phi_2 = -0.1873 (SE 0.034)    theta_2 = +0.7311 (SE 0.036)
sigma^2 = 0.0810 (SE 0.002)
```

### Step 5 — Residual diagnostics

| Test | Result | Verdict |
|---|---|---|
| Ljung-Box (residuals, lags 10/20/30) | p = 0.71 / 0.82 / 0.69 | ✓ white noise |
| Shapiro-Wilk (first 500) | W = 0.988, p ≈ 5e-4 | mild non-normality (sensitive at n≈2100) |
| **McLeod-Li** (Ljung-Box on ε²) | p ~ 1e-8 | ✗ **ARCH effect** |

The mean model is clean (Ljung-Box passes), but McLeod-Li shows volatility clustering in the squared residuals — a genuine conditional-heteroscedasticity effect, and the clearest motivation for a future ARIMA+GARCH extension.

### Step 6 — Forecasting

I report the **1-step-ahead** forecast (fixed parameters, filtered through the test set) — the honest metric for a short-horizon model, since a 530-step static forecast just reverts to the mean. Back-transform uses the log-normal bias correction `X̂ = exp(Ŷ + σ²/2) − eps`.

| Forecaster | Test MAPE |
|---|---|
| **ARIMA(2,1,2) 1-step** | **20.1%** |
| Persistence (lag 1) | 23.2% |
| Naive seasonal (S = 180) | 31.2% |

The model beats both baselines, so it adds real short-horizon information.

### Step 7 — Spectral analysis

Periodogram to locate seasonality (on the series); Fisher's g-test and the KS cumulative-periodogram test to check the **residuals** are white (both implemented from scratch — R equivalents are `GeneCycle::fisher.g.test` and `stats::cpgram`).

| Test | Statistic | p-value | Conclusion |
|---|---|---|---|
| Fisher g (series) | g = 0.078 | ~1e-44 | strong periodicity in `X_t` |
| Fisher g (residuals) | g = 0.008 | 0.23 | no leftover periodicity ✓ |
| KS cumulative periodogram (residuals) | D = 0.018 | 0.87 | residuals ~ white noise ✓ |

Dominant periodogram peaks (excluding trend harmonics): **73.6, 94.6, 66.2, 69.7 min** — a band around the ~90-minute NREM-REM cycle. The spread across periods is exactly why a single seasonal lag failed in Step 3.

---

## Results Summary

| Component | Result |
|---|---|
| Series length | 2650 epochs (~22.1 hours) |
| Train / Test | 2120 / 530 epochs (80/20) |
| Best model | ARIMA(2,1,2), AICc = 700.9 |
| Ljung-Box (residuals) | p = 0.71 / 0.82 / 0.69 — white noise ✓ |
| McLeod-Li | p ~ 1e-8 — ARCH effect (future GARCH work) |
| Forecast MAPE (1-step) | 20.1% (beats persistence 23.2%, naive 31.2%) |
| Dominant spectral band | ~66–95 min ultradian NREM-REM cycle |
| Residual Fisher / KS | p = 0.23 / 0.87 — residuals spectrally white ✓ |

---

## Figures

All figures are regenerated by `pipeline.py` into `outputs/figures/`:

`eda_plots.png` · `acf_pacf.png` · `residuals_ARIMA212.png` · `forecast.png` · `spectral_analysis.png`

---

## Repository Structure

```
eeg-sleep-sarima/
├── data/                       # PhysioNet EDF files (excluded from git)
├── outputs/figures/            # EDA, ACF/PACF, residuals, forecast, spectral
├── eeg_sleep_sarima.ipynb      # Main analysis notebook (step-by-step, executed)
├── pipeline.py                 # Headless runnable script (same logic)
├── report_outline.md           # Report draft / outline
├── requirements.txt
└── .gitignore
```

---

## Getting Started

```bash
git clone https://github.com/clarktenge/eeg-sleep-sarima.git
cd eeg-sleep-sarima
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the EDF files from [PhysioNet Sleep-EDF](https://physionet.org/content/sleep-edfx/1.0.0/) into `data/`, then:

```bash
python pipeline.py                  # regenerates every figure in outputs/figures/
# or open eeg_sleep_sarima.ipynb for the step-by-step walkthrough
```

---

## Key Findings

1. **The 90-minute seasonal model is the wrong tool here.** It over-differences, has no ACF/PACF signature, and is computationally impractical. The honest move was to drop it and capture the rhythm spectrally.
2. **ADF alone is not enough.** `log(X_t)` passes ADF yet still needs differencing — the test only sees unit-root non-stationarity, so I read it alongside the EDA and ACF.
3. **AICc disagreed with my ACF/PACF guess, and it was right.** MA(1) looked clean on the plots but failed Ljung-Box; ARIMA(2,1,2) is the AICc-best model with white-noise residuals.
4. **The residuals are linearly white but nonlinearly structured.** Ljung-Box and the spectral tests pass, but McLeod-Li reveals an ARCH effect — a clear, honest direction for future GARCH work.

---

## Data

[PhysioNet Sleep-EDF Expanded Database](https://physionet.org/content/sleep-edfx/1.0.0/). Kemp B. et al. (2000), *Electroenceph. Clin. Neurophysiol.* 69(4); Goldberger A.L. et al. (2000), *Circulation* 101(23). EDF files are excluded from version control; access requires a free PhysioNet account.

## Dependencies

Python 3.10+ · `mne` · `scipy` · `statsmodels` · `numpy` · `pandas` · `matplotlib` · `seaborn`. See `requirements.txt`.
