# Rhythms of the Sleeping Brain: ARIMA Modeling and Spectral Analysis of EEG Alpha Power Across Human Sleep Cycles

**Author:** Clark Enge
**Course:** PSTAT W 274 — Time Series Analysis
**Data:** PhysioNet Sleep-EDF Expanded Database (Kemp et al., 2000), DOI: https://doi.org/10.13026/C2X676

> This is my working outline/draft for the final report. Each section lists what I will write and the key numbers
> I will cite, all taken from my own runs in `eeg_sleep_sarima.ipynb` / `pipeline.py`.

---

## Abstract / Executive Summary

One to two paragraphs. I ask whether a classical, hand-built ARIMA model (no automated selection) can forecast
overnight EEG alpha-band power at 30-second resolution, and where the seasonal structure of sleep belongs in such
a model. I describe the data (PhysioNet Sleep-EDF, Fpz-Cz, per-epoch Welch alpha power → ~2650-point series), the
five candidate models compared by hand-coded AICc, and the winner, ARIMA(2,1,2) (AICc = 700.9). I report the key
validation numbers: Ljung-Box p = 0.71/0.82/0.69 (white-noise residuals), 1-step-ahead MAPE = 20.1% (beating
persistence 23.2% and naive-seasonal 31.2%), and the spectral result that the dominant rhythm sits in a 66–95 min
band around the ~90-minute NREM-REM cycle. I close on the one honest negative: McLeod-Li reveals an ARCH effect, so
the variance model is incomplete (future GARCH work).

---

## 1. Introduction

### 1.1 Motivation
Closed-loop neural devices need a running sleep-state estimate to calibrate stimulation (down-regulate in slow-wave
sleep, recalibrate in REM). Alpha power (8–13 Hz) is a usable proxy: suppressed in N2/N3, elevated at sleep onset,
transient in REM, and modulated by the ~90-minute ultradian cycle. A model that anticipates alpha power a step or
two ahead is a lightweight, interpretable state predictor — and a classical ARIMA is fully transparent and
algebraically auditable, which matters for any regulated setting.

### 1.2 Dataset
PhysioNet Sleep-EDF Expanded: whole-night PSG recordings, healthy subjects. I use recording `SC4001E0-PSG.edf`,
channel `EEG Fpz-Cz` (frontal midline, 100 Hz). The recording spans ~22 hours (2650 epochs of 30 s). I acknowledge
the data source explicitly and note the EDF files are not redistributed (PhysioNet credentialed access).

### 1.3 Scope and course constraints
State plainly: no automated model selection anywhere; AICc is hand-coded; all diagnostics are computed from raw
residual vectors, not wrapper functions like `checkresiduals`.

---

## 2. Methodology

### 2.1 Signal processing and feature extraction
From raw EDF to the scalar series `X_t`: (1) per-epoch 4th-order zero-phase Butterworth band-pass [8,13] Hz via
`sosfiltfilt`; (2) Welch PSD with 2 s Hann windows; (3) alpha power = trapezoidal integral of the PSD over [8,13].
Report series length (2650), units (μV²/Hz), mean (~7.2e-12), skew (right-skewed).

### 2.2 Exploratory data analysis
Three plots: raw trace, rolling mean/std (window 6 epochs), histogram. Key reading: rolling std tracks rolling
mean → variance proportional to level → multiplicative process → log transform. No strong monotone trend, but a
wandering mean.

### 2.3 Stationarity and transformation
Log transform `Y_t = log(X_t + eps)`. ADF table — and the caveat that ADF only detects a unit root:

| Series | ADF stat | p-value | Decision |
|---|---|---|---|
| `log(X_t)` | −7.69 | ~1e-11 | stationary by ADF, but mean still wanders |
| `∇ log(X_t)` | −16.92 | ~1e-29 | stationary |

Justify `d = 1`: ADF passes on the log level, but the EDA mean and slow ACF decay say one difference is still
needed for ARMA modeling — the ADF caveat made concrete.

### 2.4 The seasonal question (a documented dead-end)
I tested the physiologically obvious S = 18 / S = 180 seasonal model and rejected it on three grounds:
(1) **over-differencing** — variance rises from 0.111 to 0.230 (S=18) / 0.220 (S=180) instead of falling;
(2) **no seasonal ACF/PACF spike** at lag 18/147/180/189 (all within ±0.043);
(3) **infeasibility** — a state-space SARIMA at S=180 has a ~180-dim state and does not converge.
Conclusion: model the mean non-seasonally; the ultradian rhythm is documented spectrally (Section 4).

### 2.5 Model identification
ACF of `∇ log(X_t)`: one spike at lag 1 (−0.38), then within the band; PACF tails off → first guess MA(1) =
ARIMA(0,1,1). Carry it forward and let diagnostics + AICc test it.

### 2.6 Model fitting and selection
Hand-coded AICc = −2ℓ + 2k + 2k(k+1)/(n−k−1). Comparison table:

| Model | AICc | Ljung-Box (10/20/30) |
|---|---|---|
| ARIMA(0,1,1) | 793.4 | fails |
| ARIMA(1,1,1) | 732.6 | borderline |
| ARIMA(2,1,1) | 725.8 | passes |
| ARIMA(1,1,2) | 719.2 | passes |
| **ARIMA(2,1,2)** | **700.9** | passes (0.71/0.82/0.69) |

Answer the assignment's question directly: the AICc model is **not** the ACF/PACF guess — MA(1) fails Ljung-Box, so
AICc + diagnostics chose the richer ARIMA(2,1,2). Algebraic form with fitted values:
`(1 − 1.134B + 0.187B²)(1 − B) Y_t = (1 − 1.730B + 0.731B²) ε_t`, σ̂² = 0.081.

### 2.7 Residual diagnostics
- Residual trace, residual ACF (all in band), Q-Q plot.
- Ljung-Box (residuals): p = 0.71/0.82/0.69 → white noise.
- Shapiro-Wilk: W = 0.988, p ≈ 5e-4 → mild non-normality (test very sensitive at n≈2100; Q-Q nearly straight).
- **McLeod-Li** (Ljung-Box on ε²): p ~ 1e-8 → ARCH effect. The mean model is adequate but variance is not constant;
  this motivates a GARCH extension. I report this rather than hide it.

---

## 3. Forecasting Results

### 3.1 Back-transformation
Log-normal bias-corrected point forecast `X̂ = exp(Ŷ + σ̂²/2) − eps`; 95% interval from the inverted log bounds.
Explain why omitting σ̂²/2 underestimates the mean.

### 3.2 Accuracy
1-step-ahead (fixed parameters, filtered through the test set — the honest metric, since the static 530-step
forecast reverts to the mean):

| Forecaster | MAPE |
|---|---|
| ARIMA(2,1,2) 1-step | 20.1% |
| Persistence (lag 1) | 23.2% |
| Naive seasonal (S=180) | 31.2% |

The model beats both baselines. Caveat: McLeod-Li implies the constant-variance prediction intervals are the weak
point (too wide in calm stretches, too narrow at transitions).

---

## 4. Spectral Analysis

### 4.1 Periodogram (seasonality)
Periodogram of `log X_t` in cycles/hour. Fisher's g-test on the series rejects white noise (p ~ 1e-44). Dominant
peaks (excluding trend harmonics): 73.6, 94.6, 66.2, 69.7 min — a band around the ~90-min ultradian cycle. This
confirms the sleep architecture and explains why a single seasonal lag failed in Section 2.4.

### 4.2 Fisher and KS tests on residuals
Per the course's spectral requirement, I run Fisher's g-test and the KS cumulative-periodogram (Bartlett) test on
the ARIMA(2,1,2) residuals: Fisher g = 0.008 (p = 0.23) and KS D = 0.018 (p = 0.87) — both fail to reject, so the
residuals are spectrally white. This cross-checks the time-domain Ljung-Box result from the frequency domain. Both
tests were implemented from their definitions (R equivalents: `GeneCycle::fisher.g.test`, `stats::cpgram`).

### 4.3 Parametric AR spectral estimate
AR PSD with AIC-selected order (17 on the log series) for a smoother view of the low-frequency peaks; connect the
ultradian band to the spectral story rather than to a SARIMA seasonal term.

---

## 5. Conclusion
- **Goals met:** a hand-built ARIMA(2,1,2) forecasts alpha power with 20.1% 1-step MAPE and white-noise residuals.
- **Statistical:** residuals pass Ljung-Box and the spectral white-noise tests; they fail McLeod-Li (ARCH).
- **Domain:** the ~66–95 min spectral band confirms the NREM-REM ultradian rhythm, but it is too broad to enter the
  model as a hard seasonal period — the central methodological finding.
- **Future work:** ARIMA+GARCH for the conditional variance; multi-subject generalisation.
- Final model formula: `(1 − 1.134B + 0.187B²)(1 − B) Y_t = (1 − 1.730B + 0.731B²) ε_t`, σ̂² = 0.081.

**Acknowledgements.** PhysioNet for the data; the PSTAT W 274 instructor and TAs for guidance on diagnostics;
classmates for discussion of the over-differencing diagnosis.

---

## References
1. Kemp B., Värri A., Rosa A.C., Nielsen K.D., Gade J. (2000). A simple format for exchange of digitized polygraphic recordings. *Electroenceph. Clin. Neurophysiol.* 69(4), 391–395.
2. Goldberger A.L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23), e215–e220.
3. Box G.E.P., Jenkins G.M., Reinsel G.C., Ljung G.M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
4. Hurvich C.M., Tsai C.L. (1989). Regression and time series model selection in small samples. *Biometrika* 76(2), 297–307.
5. McLeod A.I., Li W.K. (1983). Diagnostic checking ARMA time series models using squared-residual autocorrelations. *J. Time Ser. Anal.* 4(4), 269–273.
6. Welch P.D. (1967). The use of FFT for the estimation of power spectra. *IEEE Trans. Audio Electroacoust.* 15(2), 70–73.

---

## Appendix
- **A. Code:** full `pipeline.py` with comments (and the notebook `eeg_sleep_sarima.ipynb`).
- **B. Parameter tables:** fitted coefficients with standard errors for all five candidate models.
- **C. Raw output:** console output from the ADF, Ljung-Box, McLeod-Li, Fisher, and KS calls.
