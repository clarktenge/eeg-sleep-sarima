# =============================================================================
# Rhythms of the Sleeping Brain
# ARIMA Modeling & Spectral Analysis of EEG Alpha Power Across Human Sleep Cycles
# PhysioNet Sleep-EDF Database (Kemp et al., 2000) -- channel EEG Fpz-Cz
#
# Author : Clark Enge  (clarkenge@ucsb.edu)
# Course : PSTAT W 274 -- Time Series Analysis
#
# Headless, end-to-end script. Reproduces every figure in outputs/figures/ and
# prints every number I quote in the report. Everything here is done by hand --
# no auto_arima, no checkresiduals, no automated model search.
# =============================================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # headless backend -- no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import signal
from scipy.stats import probplot, shapiro, kstest
from scipy.special import comb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AutoReg          # replaces deprecated sm.tsa.AR
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "axes.titlesize": 12})

# --- File paths ---
EDF_FILE   = "data/SC4001E0-PSG.edf"
HYPNO_FILE = "data/SC4001EC-Hypnogram.edf"

# --- Constants ---
EPOCH_SEC  = 30          # standard PSG scoring epoch (seconds)
ALPHA_LO   = 8.0         # alpha band lower edge (Hz)
ALPHA_HI   = 13.0        # alpha band upper edge (Hz)
FS         = 100         # sampling rate of the recording (Hz)
TRAIN_FRAC = 0.80
# Ultradian NREM-REM cycle is ~90 min = 180 epochs. I keep this only as a label
# for the spectral plots. I do NOT use it as a SARIMA seasonal period -- Section 6
# shows seasonal differencing at this lag over-differences the series.
S_ULTRADIAN = 180

FIGURES_DIR = "outputs/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)


# =============================================================================
# 1. EDF LOADING
# =============================================================================
def load_edf_channel(edf_path: str, channel: str = "EEG Fpz-Cz") -> tuple:
    """
    Load one EEG channel from an EDF file with MNE and resample to FS if needed.
    Returns (raw_signal: np.ndarray, sfreq: float).
    """
    import mne
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick(channel)                       # pick_channels() is deprecated in MNE >= 1.7
    if raw.info["sfreq"] != FS:
        raw.resample(FS, npad="auto")
    data, _ = raw[:]
    return data.squeeze(), raw.info["sfreq"]


# =============================================================================
# 2. BAND-PASS FILTER + WELCH ALPHA POWER EXTRACTION
# =============================================================================
def bandpass_filter(signal_arr: np.ndarray, lo: float, hi: float,
                    fs: float, order: int = 4) -> np.ndarray:
    """
    Zero-phase 4th-order Butterworth band-pass via second-order sections.
    Zero-phase (sosfiltfilt) matters on 30 s epochs: a causal filter would add a
    phase lag that distorts the per-epoch power estimate.
    """
    nyq = 0.5 * fs
    sos = signal.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return signal.sosfiltfilt(sos, signal_arr)


def compute_epoch_alpha_power(raw_signal: np.ndarray, fs: float,
                              epoch_sec: int = EPOCH_SEC,
                              lo: float = ALPHA_LO,
                              hi: float = ALPHA_HI) -> np.ndarray:
    """
    Build the univariate series X_t of alpha-band power, one value per epoch.

    For each 30 s epoch t:
      1. slice N = fs * epoch_sec samples
      2. band-pass filter to [lo, hi] Hz
      3. Welch PSD with 2 s Hann windows, 50% overlap
      4. integrate the PSD over [lo, hi]:  X_t = trapz(Pxx[alpha], f[alpha])
    """
    N_ep = int(fs * epoch_sec)
    n_epochs = len(raw_signal) // N_ep
    alpha_power = np.empty(n_epochs)
    for t in range(n_epochs):
        seg = raw_signal[t * N_ep:(t + 1) * N_ep]
        filt = bandpass_filter(seg, lo, hi, fs)
        freqs, pxx = signal.welch(filt, fs=fs, nperseg=int(fs * 2),
                                  window="hann", scaling="density")
        mask = (freqs >= lo) & (freqs <= hi)
        alpha_power[t] = np.trapezoid(pxx[mask], freqs[mask])   # np.trapz renamed in NumPy 2.x
    return alpha_power


# --- Build X_t ---
print("Loading EDF signal...")
raw_eeg, sfreq = load_edf_channel(EDF_FILE)
print("Extracting alpha power per epoch...")
X_raw = compute_epoch_alpha_power(raw_eeg, sfreq)
X_t = pd.Series(X_raw, name="alpha_power_uV2")
print(f"Series length: {len(X_t)} epochs  (~{len(X_t) * EPOCH_SEC / 3600:.1f} hours)")
print(f"Range: [{X_t.min():.3e}, {X_t.max():.3e}]  mean={X_t.mean():.3e}  skew={X_t.skew():.3f}")


# =============================================================================
# 3. TRAIN / TEST SPLIT  (80 / 20, no shuffling)
# =============================================================================
split_idx = int(len(X_t) * TRAIN_FRAC)
train = X_t.iloc[:split_idx].copy()
test  = X_t.iloc[split_idx:].copy()
print(f"Train: {len(train)} epochs | Test: {len(test)} epochs")


# =============================================================================
# 4. EDA PLOTS
# =============================================================================
def plot_eda(series: pd.Series, title_prefix: str = "Training") -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)

    axes[0].plot(series.values, color="#4C72B0", lw=0.9)
    axes[0].set_title(f"{title_prefix} Series - Alpha Power")
    axes[0].set_xlabel("Epoch (30 s)")
    axes[0].set_ylabel("Power (uV^2/Hz)")

    roll = series.rolling(window=6)          # 6 epochs = 3 min
    axes[1].plot(series.values, alpha=0.4, color="#4C72B0", lw=0.7, label="Raw")
    axes[1].plot(roll.mean().values, color="#C44E52", lw=1.5, label="Rolling mean (6 ep)")
    axes[1].plot(roll.std().values,  color="#55A868", lw=1.5, label="Rolling std (6 ep)")
    axes[1].set_title("Rolling Statistics - Trend & Variance Inspection")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=9)

    axes[2].hist(series.dropna(), bins=40, color="#4C72B0", edgecolor="white", alpha=0.8)
    axes[2].set_title("Marginal Distribution of Alpha Power")
    axes[2].set_xlabel("Power (uV^2/Hz)")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/eda_plots.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR}/eda_plots.png")


plot_eda(train)


# =============================================================================
# 5. TRANSFORMATION & STATIONARITY
# =============================================================================
def log_transform(series: pd.Series) -> pd.Series:
    """
    Variance-stabilising log transform: Y_t = log(X_t + eps),
    eps = 1e-6 * median(X_t) guards against log(0). The rolling-std plot tracks the
    rolling mean, so variance grows with level -- the multiplicative case the log fixes.
    """
    eps = 1e-6 * series.median()
    return np.log(series + eps)


def manual_difference(series: pd.Series, d: int = 1,
                      D: int = 0, S: int = 0) -> pd.Series:
    """
    Regular differencing (order d) then seasonal differencing (order D, period S).
    Regular:  (1 - B)^d Y_t
    Seasonal: (1 - B^S)^D Y_t
    """
    out = series.copy()
    for _ in range(d):
        out = out.diff(1).dropna()
    if D > 0 and S > 0:
        for _ in range(D):
            out = out.diff(S).dropna()
    return out


def run_adf_test(series: pd.Series, label: str = "") -> dict:
    """
    Augmented Dickey-Fuller test.
    Caveat I keep front of mind: ADF only tests for a UNIT ROOT in the AR part
    (a stochastic trend). It says nothing about seasonality, variance changes, or
    structural breaks -- a series can "pass" ADF and still be non-stationary.
    H0: unit root present (non-stationary). Reject if p < 0.05.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    out = {"label": label, "adf_stat": result[0], "p_value": result[1],
           "n_lags": result[2], "n_obs": result[3],
           "crit_5pct": result[4]["5%"]}
    print(f"ADF [{label}]: stat={out['adf_stat']:+.4f}  p={out['p_value']:.4g}  "
          f"lags={out['n_lags']}  crit5%={out['crit_5pct']:.3f}  "
          f"-> {'STATIONARY' if out['p_value'] < 0.05 else 'NON-STATIONARY'}")
    return out


eps_val = 1e-6 * X_t.median()
Y_t = log_transform(train)
W_t = manual_difference(Y_t, d=1)

print("\n--- Stationarity ---")
adf_log  = run_adf_test(Y_t, "log(X_t)")
adf_diff = run_adf_test(W_t, "d1 log(X_t)")


# =============================================================================
# 6. SEASONAL HEURISTIC TEST  (why I do NOT use a SARIMA seasonal term)
# -----------------------------------------------------------------------------
# Physiology says the NREM-REM cycle is ~90 min = 180 epochs, so the obvious move
# is a seasonal SARIMA at S = 180 (or S = 18 ~ 9 min). I tested that heuristic and
# rejected it on three independent grounds:
#
#   (1) OVER-DIFFERENCING: seasonal differencing should REDUCE variance. At both
#       S = 18 and S = 180 it INCREASES variance vs. the plain first difference,
#       the textbook signature of over-differencing (printed below).
#   (2) NO SEASONAL SIGNATURE: the ACF/PACF of d1 log(X_t) has no significant
#       spike at lag 18, 147, 180 or 189 (all inside the +-1.96/sqrt(n) band).
#   (3) INFEASIBILITY: a state-space SARIMA with S = 180 has a ~180-dim state and
#       does not converge in any reasonable time on this series.
#
# Conclusion: model the mean with a non-seasonal ARIMA, and treat the ultradian
# rhythm as a SPECTRAL finding (Section 11) rather than a differencing operation.
# =============================================================================
print("\n--- Seasonal heuristic: over-differencing check ---")
var_log = Y_t.var()
var_d1  = W_t.var()
print(f"  var[log(X_t)]          = {var_log:.5f}")
print(f"  var[d1 log(X_t)]       = {var_d1:.5f}")
for S in (18, S_ULTRADIAN):
    var_sd = manual_difference(Y_t, d=1, D=1, S=S).var()
    flag = "INCREASED -> over-differencing" if var_sd > var_d1 else "reduced -> justified"
    print(f"  var[d1 D1_S{S:<3} log(X_t)] = {var_sd:.5f}   ({flag})")

# Quick numeric scan of ACF/PACF at candidate seasonal lags
W_full = W_t.dropna()
ci_band = 1.96 / np.sqrt(len(W_full))
a_seas = acf(W_full, nlags=200)
p_seas = pacf(W_full, nlags=200, method="ywm")
print(f"  seasonal-lag ACF/PACF (95% band = +-{ci_band:.3f}):")
for k in (18, 147, 180, 189):
    print(f"    lag {k:>3}: ACF={a_seas[k]:+.3f}  PACF={p_seas[k]:+.3f}")


# =============================================================================
# 7. ACF & PACF -- MODEL IDENTIFICATION
# =============================================================================
def plot_acf_pacf(series: pd.Series, lags: int = 40,
                  title: str = "", fname: str = "acf_pacf.png") -> None:
    """
    Side-by-side ACF and PACF with 95% bands.
    Identification rules I apply by eye:
      ACF cuts off at q, PACF tails off -> MA(q)
      PACF cuts off at p, ACF tails off -> AR(p)
      both tail off                     -> mixed ARMA(p,q)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(),  lags=lags, alpha=0.05, ax=ax1,
             title=f"ACF - {title}", zero=False)
    plot_pacf(series.dropna(), lags=lags, alpha=0.05, ax=ax2,
              title=f"PACF - {title}", zero=False, method="ywm")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{fname}", bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR}/{fname}")


# d1 log(X_t): ACF cuts off after lag 1, PACF tails off -> first guess MA(1) = ARIMA(0,1,1)
plot_acf_pacf(W_t, lags=40, title="d1 log(X_t)", fname="acf_pacf.png")


# =============================================================================
# 8. AICc + SARIMA FITTING
# =============================================================================
def compute_aicc(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """
    AICc = -2 l(theta) + 2k + 2k(k+1)/(n-k-1).  k counts sigma^2.
    Preferred over AIC when n/k is not large.
    """
    if n_obs - n_params - 1 <= 0:
        raise ValueError(f"Overparameterised: n={n_obs}, k={n_params}")
    aic  = -2.0 * log_likelihood + 2.0 * n_params
    corr = (2.0 * n_params * (n_params + 1)) / (n_obs - n_params - 1)
    return aic + corr


def fit_arima(endog: pd.Series, order: tuple, label: str,
              verbose: bool = True) -> dict:
    """
    Fit a (non-seasonal) ARIMA via SARIMAX and report stats by hand.
    Model: phi_p(B) (1-B)^d Y_t = theta_q(B) eps_t,  eps_t ~ WN(0, sigma^2).
    """
    model  = SARIMAX(endog, order=order, seasonal_order=(0, 0, 0, 0),
                     trend="n", enforce_stationarity=True, enforce_invertibility=True)
    result = model.fit(disp=False, maxiter=200)
    n_obs    = result.nobs
    n_params = result.df_model + 1            # +1 for sigma^2
    aicc_val = compute_aicc(result.llf, n_params, n_obs)
    if verbose:
        print(f"  {label:<13} ll={result.llf:8.2f}  AIC={result.aic:8.2f}  "
              f"AICc={aicc_val:8.2f}  BIC={result.bic:8.2f}  k={n_params}")
    return {"label": label, "result": result, "aicc": aicc_val, "order": order}


print("\n--- Candidate models (log scale, d=1) ---")
candidates = {
    "ARIMA(0,1,1)": (0, 1, 1),   # the ACF/PACF first guess (MA(1))
    "ARIMA(1,1,1)": (1, 1, 1),
    "ARIMA(2,1,1)": (2, 1, 1),
    "ARIMA(1,1,2)": (1, 1, 2),
    "ARIMA(2,1,2)": (2, 1, 2),
}
fits = {name: fit_arima(Y_t, order, name) for name, order in candidates.items()}
best = min(fits.values(), key=lambda m: m["aicc"])
print(f"\nBest by AICc: {best['label']}  (AICc = {best['aicc']:.2f})")

# Print the winning coefficient table in full
res_best = best["result"]
print(f"\n{best['label']} fitted coefficients:")
for name, val, se in zip(res_best.param_names, res_best.params, res_best.bse):
    print(f"    {name:<10} = {val:+.5f}  (SE={se:.5f})")


# =============================================================================
# 9. RESIDUAL DIAGNOSTICS
# =============================================================================
def mcleod_li_test(resid: np.ndarray, lags=(10, 20, 30)) -> pd.DataFrame:
    """
    McLeod-Li test = Ljung-Box on the SQUARED residuals. Detects nonlinear
    structure / conditional heteroscedasticity (ARCH effects) that the linear
    Ljung-Box on the raw residuals cannot see.
    H0: no autocorrelation in eps_t^2 (linear model adequate). Reject -> consider GARCH.
    """
    lb = acorr_ljungbox(pd.Series(resid ** 2), lags=list(lags), return_df=True)
    print("  McLeod-Li (Ljung-Box on squared residuals):")
    print(lb.to_string().replace("\n", "\n    "))
    return lb


def residual_diagnostics(fit_result, label: str = "", fname_prefix: str = "") -> dict:
    # drop the first two values -- diffuse state-space initialisation transient
    resid = fit_result.resid.dropna().values[2:]

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(resid, color="#4C72B0", lw=0.8)
    ax1.axhline(0, color="red", lw=0.8, ls="--")
    ax1.set_title(f"Residuals - {label}")
    ax1.set_xlabel("Epoch")
    ax2 = fig.add_subplot(gs[1, 0])
    plot_acf(resid, lags=40, alpha=0.05, ax=ax2, zero=False, title="ACF of Residuals")
    ax3 = fig.add_subplot(gs[1, 1])
    probplot(resid, dist="norm", plot=ax3)
    ax3.set_title("Q-Q Plot of Residuals")
    plt.tight_layout()
    fname = f"{FIGURES_DIR}/residuals_{fname_prefix}.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

    print(f"\nResidual tests ({label}):")
    lb = acorr_ljungbox(resid, lags=[10, 20, 30], return_df=True)
    print("  Ljung-Box (residuals):")
    print(lb.to_string().replace("\n", "\n    "))
    sw_stat, sw_p = shapiro(resid[:500])
    print(f"  Shapiro-Wilk (first 500): W={sw_stat:.4f}  p={sw_p:.2e}")
    ml = mcleod_li_test(resid, lags=(10, 20, 30))
    return {"resid": resid, "ljungbox": lb, "mcleodli": ml}


diag = residual_diagnostics(res_best, label=best["label"], fname_prefix="ARIMA212")


# =============================================================================
# 10. FORECASTING & BACK-TRANSFORMATION
# =============================================================================
def backtransform_mean(log_mean, log_var, eps):
    """Log-normal bias-corrected mean: E[X] = exp(mu + sigma^2/2) - eps."""
    return np.exp(log_mean + log_var / 2.0) - eps


def one_step_ahead(fit_dict, full_series, split, eps):
    """
    Genuine 1-step-ahead forecasts over the test window using the FIXED fitted
    parameters (state-space filtering via .append(..., refit=False)). This is the
    honest evaluation for a short-horizon model -- a 530-step static forecast just
    reverts to the mean and is meaningless here.
    """
    y_test_log = np.log(full_series.iloc[split:] + eps)
    appended   = fit_dict["result"].append(y_test_log, refit=False)
    pred_log   = appended.predict(start=split, end=len(full_series) - 1)
    sigma2     = fit_dict["result"].params[-1]
    return backtransform_mean(pred_log.values, sigma2, eps)


def static_forecast(fit_dict, steps, eps):
    """Static multi-step forecast on the log scale, back-transformed (for the plot/CI)."""
    fc      = fit_dict["result"].get_forecast(steps=steps)
    fc_mean = fc.predicted_mean.values
    fc_var  = fc.var_pred_mean.values
    ci      = fc.conf_int(alpha=0.05).values
    df = pd.DataFrame({
        "forecast": backtransform_mean(fc_mean, fc_var, eps),
        "lower_95": np.exp(ci[:, 0]) - eps,
        "upper_95": np.exp(ci[:, 1]) - eps,
    })
    return df


def accuracy(actual, pred, label):
    mae  = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    print(f"  {label:<22} MAE={mae:.3e}  RMSE={rmse:.3e}  MAPE={mape:.2f}%")
    return {"mae": mae, "rmse": rmse, "mape": mape}


print("\n--- Forecast evaluation (test set) ---")
actual = test.values
pred_1step = one_step_ahead(best, X_t, split_idx, eps_val)
acc_model  = accuracy(actual, pred_1step, f"{best['label']} 1-step")
# Baselines for context
accuracy(actual, X_t.shift(1).iloc[split_idx:].values,         "persistence (lag 1)")
accuracy(actual, X_t.shift(S_ULTRADIAN).iloc[split_idx:].values, "naive seasonal (S=180)")

fc_df = static_forecast(best, steps=len(test), eps=eps_val)


def plot_forecast(train_orig, test_orig, pred_1step, fc_df, label=""):
    fig, ax = plt.subplots(figsize=(16, 5))
    n_tr, n_te = len(train_orig), len(test_orig)
    t_tr = np.arange(n_tr)
    t_te = np.arange(n_tr, n_tr + n_te)
    ax.plot(t_tr, train_orig.values, color="#4C72B0", lw=0.8, label="Training")
    ax.plot(t_te, test_orig.values,  color="#55A868", lw=0.8, label="Actual (test)")
    ax.plot(t_te, pred_1step, color="#C44E52", lw=1.0,
            label=f"1-step forecast ({label})")
    ax.fill_between(t_te, fc_df["lower_95"].values, fc_df["upper_95"].values,
                    color="#8172B3", alpha=0.18, label="95% PI (static)")
    ax.set_title(f"Alpha Power Forecast - {label}")
    ax.set_xlabel("Epoch (30 s)")
    ax.set_ylabel("Alpha Power (uV^2/Hz)")
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/forecast.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR}/forecast.png")


plot_forecast(train, test, pred_1step, fc_df, label=best["label"])


# =============================================================================
# 11. SPECTRAL ANALYSIS
#   - periodogram on the SERIES (to locate the seasonality)
#   - Fisher g-test + KS cumulative-periodogram test on the RESIDUALS
#     (white-noise check, per the spectral requirement)
#   - parametric AR PSD for a smoother peak
#
# Fisher's g-test and the KS cumulative-periodogram (Bartlett) test are not in
# statsmodels/scipy as one-liners; I implemented both from their definitions.
# (R equivalents: GeneCycle::fisher.g.test and stats::cpgram.)
# =============================================================================
def fisher_g_test(Pxx: np.ndarray) -> tuple:
    """
    Fisher g-test: is the largest periodogram ordinate larger than white noise allows?
        g = max(I_j) / sum(I_j)
        p = sum_{k=1}^{floor(1/g)} (-1)^{k-1} C(m,k) (1 - k g)^{m-1}
    H0: white noise (flat spectrum). Reject (p<0.05) -> significant periodicity.
    """
    m = len(Pxx)
    g = float(np.max(Pxx) / np.sum(Pxx))
    upper = int(np.floor(1.0 / g))
    p_val = sum((-1) ** (k - 1) * float(comb(m, k, exact=True)) * (1 - k * g) ** (m - 1)
                for k in range(1, upper + 1))
    return g, float(np.clip(p_val, 0.0, 1.0))


def ks_cumulative_periodogram(Pxx: np.ndarray) -> tuple:
    """
    Bartlett's KS test on the normalised cumulative periodogram
        C(k) = sum_{j<=k} I_j / sum_j I_j
    Under white noise the C(k) behave like ordered Uniform(0,1) draws, so a KS test
    against Uniform(0,1) checks for leftover spectral structure. (Drop the final
    point, which is identically 1.)  Reject (p<0.05) -> structure remains.
    """
    cum = np.cumsum(Pxx) / np.sum(Pxx)
    ks_stat, p_val = kstest(cum[:-1], "uniform")
    return float(ks_stat), float(p_val), cum


def spectral_analysis(series_full: np.ndarray, residuals: np.ndarray,
                      epoch_sec: int = EPOCH_SEC, ar_max_order: int = 25) -> dict:
    dt_hr = epoch_sec / 3600.0

    # --- periodogram of the (log) series -> locate seasonality ---
    f_ep, Pxx = signal.periodogram(series_full, fs=1.0)   # fs = 1 / epoch
    f_hr   = f_ep[1:] / dt_hr                              # cycles per hour (drop DC)
    Pxx_nz = Pxx[1:]

    g_ser, gp_ser = fisher_g_test(Pxx_nz)
    print("\n--- Spectral analysis ---")
    print(f"  Periodogram of series : Fisher g={g_ser:.5f} p={gp_ser:.3g}  "
          "(strong periodicity expected)")
    print("  Top periodogram peaks:")
    for idx in np.argsort(Pxx_nz)[-6:][::-1]:
        cyc = f_hr[idx]
        per_min = 60.0 / cyc if cyc > 0 else np.inf
        print(f"    {cyc:6.3f} cyc/hr  ~ {per_min:7.1f} min")

    # --- Fisher + KS on RESIDUALS (white-noise check) ---
    f_r, Pxx_r = signal.periodogram(residuals, fs=1.0)
    Pxx_r = Pxx_r[1:]
    g_res, gp_res = fisher_g_test(Pxx_r)
    ks_res, ksp_res, cum_res = ks_cumulative_periodogram(Pxx_r)
    print(f"  Residual Fisher g-test : g={g_res:.5f}  p={gp_res:.3g}  "
          f"-> {'periodicity remains' if gp_res < 0.05 else 'no leftover periodicity'}")
    print(f"  Residual KS test       : D={ks_res:.4f}  p={ksp_res:.3g}  "
          f"-> {'structure remains' if ksp_res < 0.05 else 'consistent with white noise'}")

    # --- parametric AR PSD (AIC-selected order) ---
    best_aic, best_ar = np.inf, 1
    for p in range(1, ar_max_order + 1):
        try:
            m = AutoReg(series_full, lags=p, old_names=False).fit()
            if m.aic < best_aic:
                best_aic, best_ar = m.aic, p
        except Exception:
            pass
    ar_fit = AutoReg(series_full, lags=best_ar, old_names=False).fit()
    ar_params = ar_fit.params[1:]               # drop intercept
    sigma2 = np.var(ar_fit.resid)
    print(f"  Parametric AR order (AIC) = {best_ar}")
    w = np.linspace(0, np.pi, 512)
    H = np.ones(len(w), dtype=complex)
    for k, a_k in enumerate(ar_params, start=1):
        H -= a_k * np.exp(-1j * w * k)
    S_ar = sigma2 / (np.abs(H) ** 2)
    f_ar = (w / (2 * np.pi)) / dt_hr

    # --- plots ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    axes[0].semilogy(f_hr, Pxx_nz, color="#4C72B0", lw=0.8)
    axes[0].axvline(1 / 1.5, color="red", lw=1.2, ls="--",
                    label="~90-min ultradian band (0.67 cyc/hr)")
    axes[0].set_title("Periodogram of log Alpha-Power Series")
    axes[0].set_xlabel("Frequency (cycles / hour)")
    axes[0].set_ylabel("Power (log scale)")
    axes[0].set_xlim(0, 8)
    axes[0].legend(fontsize=9)

    axes[1].semilogy(f_ar[1:], S_ar[1:], color="#C44E52", lw=1.0)
    axes[1].set_title(f"Parametric AR({best_ar}) Spectral Estimate (AIC-selected)")
    axes[1].set_xlabel("Frequency (cycles / hour)")
    axes[1].set_ylabel("PSD")
    axes[1].set_xlim(0, 8)

    freq_pos = np.linspace(0, 1, len(cum_res))
    axes[2].plot(freq_pos, cum_res, color="#4C72B0", lw=1.2,
                 label="Cumulative periodogram (residuals)")
    axes[2].plot(freq_pos, freq_pos, color="red", lw=1.0, ls="--",
                 label="Expected under white noise")
    axes[2].set_title(f"Residual Cumulative Periodogram  (KS D={ks_res:.4f}, p={ksp_res:.3g})")
    axes[2].set_xlabel("Normalised frequency")
    axes[2].set_ylabel("Cumulative spectral mass")
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/spectral_analysis.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR}/spectral_analysis.png")
    return {"fisher_series": (g_ser, gp_ser), "fisher_resid": (g_res, gp_res),
            "ks_resid": (ks_res, ksp_res), "ar_order": best_ar}


spec = spectral_analysis(log_transform(X_t).values, diag["resid"], epoch_sec=EPOCH_SEC)


# =============================================================================
# 12. RESULTS SUMMARY
# =============================================================================
print("\n" + "=" * 64)
print("RESULTS SUMMARY")
print("=" * 64)
print(f"Series           : {len(X_t)} epochs (~{len(X_t)*EPOCH_SEC/3600:.1f} h)  "
      f"train={len(train)} test={len(test)}")
print(f"Best model       : {best['label']}  AICc={best['aicc']:.2f}")
print(f"1-step MAPE      : {acc_model['mape']:.2f}%")
print(f"Ljung-Box p (resid)  : "
      f"{[round(p,3) for p in diag['ljungbox']['lb_pvalue']]}  (>0.05 = white noise)")
print(f"McLeod-Li p (resid^2): "
      f"{[f'{p:.2e}' for p in diag['mcleodli']['lb_pvalue']]}  (<0.05 = ARCH effect)")
print(f"Residual Fisher  : p={spec['fisher_resid'][1]:.3g}   "
      f"Residual KS: p={spec['ks_resid'][1]:.3g}")
print("=" * 64)
print("Pipeline complete. Figures in outputs/figures/.")
