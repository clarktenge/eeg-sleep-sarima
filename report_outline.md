# Rhythms of the Sleeping Brain: SARIMA Modeling and Spectral Analysis of EEG Alpha Power Across Human Sleep Cycles

**Author:** [Your Name]
**Course:** [Time Series Analysis — Course Number]
**Instructor:** [Instructor Name]
**Date:** [Submission Date]
**Data Repository:** PhysioNet Sleep-EDF Database (Kemp et al., 2000)
DOI: https://doi.org/10.13026/C2X676

---

## Abstract / Executive Summary

> **[Orchestrator — Content Guidance]**
> Open with the applied BCI engineering hook: adaptive neural interfaces (e.g., closed-loop neuromodulation systems) must continuously estimate a patient's sleep stage to calibrate stimulation thresholds. State the core statistical question: can a classical SARIMA model, fitted entirely without automated selection tools, produce statistically valid 20%-horizon forecasts of EEG alpha-band power at 30-second resolution?
>
> Summarise in ≤250 words:
> - Data source and preprocessing (PhysioNet Sleep-EDF, Fpz-Cz, Welch epoch power)
> - The two competing SARIMA models and the winning specification (by AICc)
> - Key validation metrics: AICc differential, Ljung-Box p-values at lags 10/20/30, RMSE/MAPE on held-out test set
> - The dominant spectral peak identified (expected: ~90-minute ultradian rhythm at ~0.67 cycles/hour)
> - One-sentence conclusion on forecast quality and its BCI relevance

---

## 1. Introduction

### 1.1 Motivation: Neural Interfaces and Sleep-State Estimation

> **[Orchestrator — Domain Framing]**
> Establish the applied context before any statistics. Describe why real-time sleep-stage estimation matters for adaptive devices:
> - Closed-loop deep brain stimulators and cortical implants (cite Medtronic Percept, Neuralink research roadmap) must down-regulate stimulation during slow-wave sleep and recalibrate during REM.
> - Alpha power (8–13 Hz) is a canonical biomarker: suppressed in slow-wave sleep (N2/N3), transiently elevated at sleep onset (N1) and during REM, cycling with the ultradian ~90-minute NREM-REM architecture.
> - A forecasting model that anticipates alpha power trajectories one to three epochs ahead (30–90 seconds) could serve as a lightweight, interpretable state predictor in resource-constrained implanted hardware.
>
> Transition to the statistical framing: classical SARIMA provides a fully transparent, algebraically auditable model — critical for regulatory submissions and reproducibility in medical devices.

### 1.2 Dataset Description

> **[Orchestrator — Data Provenance Guidance]**
> Describe the PhysioNet Sleep-EDF Expanded dataset precisely:
> - 197 whole-night polysomnographic recordings, healthy subjects aged 25–101
> - Channels relevant here: EEG Fpz-Cz (frontal midline, referential), sampled at 100 Hz
> - Gold-standard hypnogram annotations (Rechtschaffen & Kales, 1968): W, N1, N2, N3, REM at 30-second resolution
> - Selection rationale: single subject (SC4001), lights-off to lights-on segment only (~8.5 hours → ~1020 epochs); justify exclusion of first/last 5 epochs (transition artifacts)
>
> Cite: Goldberger AL et al., PhysioBank, PhysioToolkit, and PhysioNet (Circulation, 2000).

### 1.3 Scope and Course Constraints

State explicitly that: (i) no automated model selection is used anywhere in this project; (ii) all model comparison is conducted via manually programmed AICc; (iii) all diagnostic plots are produced from raw residual vectors, not wrapper functions.

---

## 2. Methodology

### 2.1 Signal Processing and Feature Extraction

> **[Orchestrator — Mathematical Detail Guidance]**
> Describe the pipeline from raw EDF to the scalar time series $X_t$:
>
> **Step 1 — Band-pass filtering.** A 4th-order zero-phase Butterworth filter with passband [8, 13] Hz is applied to each 30-second epoch using second-order sections (`sosfiltfilt`) to prevent phase distortion.
>
> **Step 2 — Welch PSD estimation.** For epoch $t$ with $N = 3000$ samples (100 Hz × 30 s), Welch's method with a 2-second Hann-windowed segment computes the one-sided PSD $\hat{P}_{xx}(f)$. Alpha power is the trapezoidal integral:
>
> $$X_t = \int_{8}^{13} \hat{P}_{xx}(f)\, df \approx \sum_{f_k \in [8,13]} \hat{P}_{xx}(f_k)\, \Delta f$$
>
> Report the resulting $X_t$ series length ($\approx 900$–$1020$ epochs), units (μV²/Hz), and descriptive statistics (mean, SD, skewness, kurtosis).

### 2.2 Exploratory Data Analysis

> **[Orchestrator — EDA Interpretation Blueprint]**
> Direct the reader to interpret the three EDA plots:
> 1. **Raw series plot:** identify approximate sleep-cycle macrostructure visually (troughs ≈ slow-wave stages, peaks ≈ REM/wake transitions).
> 2. **Rolling statistics:** assess whether variance is proportional to level (suggesting a multiplicative/log model) or constant (additive model appropriate).
> 3. **Marginal histogram:** note right-skew typical of power spectral quantities → motivates log transformation.

### 2.3 Stationarity Testing and Transformation

> **[Orchestrator — Testing Protocol Guidance]**
> Present results in a structured table:
>
> | Series | ADF Statistic | p-value | 5% Critical | Conclusion |
> |---|---|---|---|---|
> | $X_t$ (raw) | | | −2.86 | Non-stationary |
> | $Y_t = \log(X_t)$ | | | −2.86 | Non-stationary (variance stabilised) |
> | $\nabla Y_t$ | | | −2.86 | Stationary |
> | $\nabla \nabla_{18} Y_t$ | | | −2.86 | Stationary |
>
> Justify the choice of $d=1$, $D=1$, $S=18$ (18 × 30 s = 9 minutes — or revisit to $S=180$ for the 90-minute cycle; discuss the trade-off in degrees of freedom lost).

### 2.4 Model Identification

> **[Orchestrator — Visual Cutoff Rules]**
> After presenting the ACF/PACF plots of $\nabla \nabla_{18} \log(X_t)$:
> - Non-seasonal lags 1–5: describe the cutoff pattern. If ACF cuts off at lag 1 → MA(1); if PACF cuts off at lag 1 → AR(1); tailing in both → mixed ARMA.
> - Seasonal lags at multiples of $S$: a single significant spike in ACF at lag $S$ identifies a seasonal MA(1) component; a single PACF spike at $S$ identifies a seasonal AR(1) component.
> - Tabulate two candidate model structures and their identified $(p, d, q)(P, D, Q)_S$ before fitting.

### 2.5 Model Fitting and Selection

> **[Orchestrator — AICc Comparison Blueprint]**
> Present the full fitted parameter table for both models: AR coefficients ($\phi_1, \ldots, \phi_p$), MA coefficients ($\theta_1, \ldots, \theta_q$), seasonal counterparts, and $\hat{\sigma}^2_\varepsilon$.
>
> Write out the **exact algebraic form** of the winning model. Example for SARIMA(1,1,1)(1,1,1)[18]:
>
> $$(1 - \Phi_1 B^{18})(1 - \phi_1 B)(1 - B^{18})(1 - B)\, Y_t = (1 + \Theta_1 B^{18})(1 + \theta_1 B)\, \varepsilon_t$$
>
> where $B$ is the backshift operator, $Y_t = \log(X_t + \varepsilon)$, and $\varepsilon_t \sim \mathrm{WN}(0, \hat{\sigma}^2)$.
>
> **Substitute the fitted numerical values** of $\phi_1, \theta_1, \Phi_1, \Theta_1, \hat{\sigma}^2$ into this expression.
>
> Model comparison table:
>
> | Model | $k$ | $n$ | Log-Lik | AIC | **AICc** | BIC |
> |---|---|---|---|---|---|---|
> | SARIMA(1,1,1)(1,1,1)[18] | | | | | | |
> | SARIMA(2,1,1)(0,1,1)[18] | | | | | | |
>
> State the AICc formula explicitly as shown in `pipeline.py` and interpret the magnitude of the difference (rule of thumb: $\Delta$AICc > 2 = meaningful; > 10 = decisive).

### 2.6 Residual Diagnostics

> **[Orchestrator — Diagnostic Checklist (No Automated Wrappers)]**
> Present each diagnostic with its null hypothesis and conclusion:
> 1. **Residual time plot:** visually assess homoscedasticity and zero-mean centering.
> 2. **ACF of residuals:** all spikes within $\pm 1.96/\sqrt{n}$ bands → no autocorrelation structure remains.
> 3. **Q-Q plot:** linearity indicates approximate Gaussianity.
> 4. **Ljung-Box test** at lags 10, 20, 30:
>    $H_0$: no autocorrelation up to lag $m$.  Tabulate $Q_m$ and p-values; retain $H_0$ for valid model.
> 5. **Shapiro-Wilk** (on first 500 residuals): $H_0$: residuals ~ Normal. Report W-statistic and p-value.
>
> If any diagnostic fails, discuss which model assumption is violated and propose a remedy (e.g., GARCH for heteroscedastic residuals in power signals).

---

## 3. Forecasting Results

### 3.1 Back-Transformation and Confidence Intervals

> **[Orchestrator — Back-Transform Math]**
> State the bias-corrected inverse transformation explicitly:
>
> Since $Y_t = \log(X_t + \varepsilon)$ and $\hat{Y}_{n+h}$ is a Gaussian forecast with variance $\hat{\sigma}^2_h$:
>
> $$\hat{X}_{n+h} = \exp\!\left(\hat{Y}_{n+h} + \frac{\hat{\sigma}^2_h}{2}\right) - \varepsilon \quad \text{(log-normal mean, bias-corrected)}$$
>
> 95% prediction interval:
> $$\left[\exp(\hat{Y}_{n+h} - 1.96\,\hat{\sigma}_h) - \varepsilon,\quad \exp(\hat{Y}_{n+h} + 1.96\,\hat{\sigma}_h) - \varepsilon\right]$$
>
> Explain why the naive back-transform $\exp(\hat{Y}_{n+h})$ underestimates the mean by a factor of $e^{\hat{\sigma}^2_h/2}$.

### 3.2 Forecast Accuracy

> Present the accuracy table for the 20% holdout:
>
> | Metric | Value | Interpretation |
> |---|---|---|
> | MAE | | Mean absolute error in μV²/Hz |
> | RMSE | | RMS error in μV²/Hz |
> | MAPE | | Percentage error relative to actual |
>
> Contextualise: what MAPE threshold would be acceptable for a BCI state estimator? (Suggest literature comparison if available.)

---

## 4. Spectral Analysis

### 4.1 Non-Parametric Periodogram

> **[Orchestrator — Spectral Interpretation Guidance]**
> Describe the x-axis in cycles per hour and annotate the following expected peaks:
> - **~0.67 cycles/hour (period ≈ 90 min):** the ultradian NREM-REM oscillation — the dominant slow modulator of alpha power. If visible, this validates the $S=18$ (or $S=180$) seasonal period choice.
> - **~6.7 cycles/hour (period ≈ 9 min):** possible sleep-spindle burst cadence (12–15 Hz spindles cluster with alpha fluctuations).
> - **~0.017 cycles/hour (period ≈ 8.5 hr):** the overnight trend if the recording shows progressive sleep deepening.
>
> Discuss periodogram leakage: why the raw periodogram is noisy, and why the parametric AR estimate provides smoother peak localisation.

### 4.2 Parametric AR Spectral Estimate

> Describe how the AR(p) order is selected (AIC over lags 1–25) and how the spectral estimate $\hat{S}_{AR}(\omega)$ sharpens the 90-minute peak. Connect the dominant spectral frequency to the seasonal period $S$ chosen in the SARIMA model — this is the confirmatory evidence linking the spectral and time-domain analyses.

---

## 5. Conclusion

> **[Orchestrator — Conclusion Framework]**
> Address three layers:
>
> 1. **Statistical:** Did the winning SARIMA model produce white-noise residuals (Ljung-Box p > 0.05) and Gaussian errors (Shapiro-Wilk p > 0.05)? Were the forecasts within acceptable MAPE bounds?
> 2. **Domain:** Does the spectral peak at ~90 minutes confirm the seasonal period embedded in the SARIMA structure? What does the alpha-power forecast trajectory reveal about the subject's sleep architecture during the test window?
> 3. **BCI/Engineering relevance:** How could this pipeline be operationalised on streaming EEG in an implanted device? What are the computational constraints (model order, update frequency), and how does a classical SARIMA compare to a Kalman filter or RNN baseline in interpretability vs. accuracy?

---

## References

1. Kemp B, Värri A, Rosa AC, Nielsen KD, Gade J. (2000). A simple format for exchange of digitized polygraphic recordings. *Electroencephalography and Clinical Neurophysiology*, 69(4), 391–395.
2. Goldberger AL, Amaral LAN, Glass L, et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215–e220. https://doi.org/10.13026/C2X676
3. Rechtschaffen A, Kales A. (1968). *A Manual of Standardized Terminology, Techniques and Scoring System for Sleep Stages of Human Subjects*. US Government Printing Office.
4. Box GEP, Jenkins GM, Reinsel GC, Ljung GM. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
5. Hurvich CM, Tsai CL. (1989). Regression and time series model selection in small samples. *Biometrika*, 76(2), 297–307.
6. Welch PD. (1967). The use of fast Fourier transform for the estimation of power spectra. *IEEE Transactions on Audio and Electroacoustics*, 15(2), 70–73.
7. [Add any additional course-assigned readings here]

---

## Appendix A: Annotated Code

> Attach the full `pipeline.py` with section headers preserved. Ensure every non-obvious code block has a one-line comment referencing the mathematical expression it implements (e.g., `# AICc = AIC + 2k(k+1)/(n-k-1)`).

## Appendix B: Model Parameter Tables

> Full coefficient tables (with standard errors and z-statistics) from `result.summary()` for both fitted models — printed as plain text, not using any automated diagnostic wrapper.

## Appendix C: Raw ADF and Ljung-Box Output

> Paste the exact console output from `run_adf_test()` and `acorr_ljungbox()` calls to demonstrate manual execution.
