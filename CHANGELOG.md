# Changelog

## [1.0] — Final

### Decided
- **Dropped the seasonal SARIMA term; final model is non-seasonal ARIMA(2,1,2).**
  I tested the physiologically motivated seasonal period (S = 18 and S = 180) and rejected it on three grounds:
  (1) seasonal differencing *increases* variance (0.111 → 0.230 at S=18 / 0.220 at S=180) — over-differencing;
  (2) the ACF/PACF of `∇ log(X_t)` has no significant spike at lag 18/147/180/189; and
  (3) a state-space SARIMA at S=180 has a ~180-dim state and does not converge in reasonable time.
  The ~90-min ultradian rhythm is spectrally broad (66–95 min), so it is documented via spectral analysis instead.

- **Model selection.** Compared ARIMA(0,1,1)/(1,1,1)/(2,1,1)/(1,1,2)/(2,1,2) by hand-coded AICc.
  ARIMA(2,1,2) wins (AICc = 700.9; ΔAICc ≈ 18 over next-best) with white-noise residuals
  (Ljung-Box p = 0.71/0.82/0.69). The AICc model is *not* the ACF/PACF guess (MA(1) failed Ljung-Box).

- **Forecast evaluation: static → 1-step-ahead.** The 530-step static forecast reverts to the mean and is
  meaningless here; I report 1-step-ahead MAPE (20.1%), which beats persistence (23.2%) and naive-seasonal (31.2%).

### Fixed
- **Spectral tests now run on the residuals.** Fisher's g-test and the KS cumulative-periodogram test are applied
  to the ARIMA(2,1,2) residuals (white-noise check: p = 0.23 / 0.87), while the periodogram is used on the series
  to locate seasonality. Previously both tests were run on the raw series.
- `np.trapz` → `np.trapezoid` (NumPy 2.0 rename).
- `raw.pick_channels([channel])` → `raw.pick(channel)` (deprecated in MNE ≥ 1.7).
- Residual diagnostics now drop the first two state-space initialisation values, removing the spurious
  initial-transient outlier that had been failing Shapiro-Wilk (W is now 0.988, not ~0.17).

### Added
- McLeod-Li test (Ljung-Box on squared residuals): p ~ 1e-8 → an ARCH effect remains. Mean model is adequate but
  the conditional variance is not constant; flagged as future ARIMA+GARCH work.
