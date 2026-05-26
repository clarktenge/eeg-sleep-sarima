# =============================================================================
# Rhythms of the Sleeping Brain: SARIMA Modeling & Spectral Analysis
# EEG Alpha Power Across Human Sleep Cycles
# PhysioNet Sleep-EDF Database — Fpz-Cz Channel
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import signal
from scipy.stats import probplot, shapiro, kstest
from scipy.special import comb
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AutoReg          # replaces deprecated sm.tsa.AR
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
import os

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "axes.titlesize": 12})

# --- File paths ---
EDF_FILE   = "data/SC4001E0-PSG.edf"
HYPNO_FILE = "data/SC4001EC-Hypnogram.edf"

EPOCH_SEC    = 30
ALPHA_LO     = 8.0
ALPHA_HI     = 13.0
FS           = 100
TRAIN_FRAC   = 0.80
S            = 18      # candidate seasonal period in epochs (~9 min); revisit after ACF/PACF

FIGURES_DIR = "outputs/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)


# =============================================================================
# 1. EDF LOADING
# =============================================================================
def load_edf_channel(edf_path: str, channel: str = "EEG Fpz-Cz") -> tuple:
    """
    Load a single EEG channel from an EDF file via MNE and resample to FS.
    Returns (raw_signal: np.ndarray, sfreq: float).
    """
    import mne
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.pick_channels([channel])
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
    Zero-phase 4th-order Butterworth band-pass filter using second-order
    sections (sosfiltfilt) to avoid phase distortion on short EEG epochs.
    """
    nyq = 0.5 * fs
    sos = signal.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return signal.sosfiltfilt(sos, signal_arr)


def compute_epoch_alpha_power(raw_signal: np.ndarray, fs: float,
                               epoch_sec: int = EPOCH_SEC,
                               lo: float = ALPHA_LO,
                               hi: float = ALPHA_HI) -> np.ndarray:
    """
    Construct the univariate time series X_t of alpha-band power.

    For each 30-second epoch t:
      1. Slice N_ep = fs * epoch_sec samples
      2. Band-pass filter to [lo, hi] Hz
      3. Welch PSD: 2-second Hann windows, 50% overlap
      4. Integrate PSD over [lo, hi]:  X_t = trapz(Pxx[alpha], f[alpha])
    """
    N_ep = int(fs * epoch_sec)
    n_epochs = len(raw_signal) // N_ep
    alpha_power = np.empty(n_epochs)

    for t in range(n_epochs):
        seg = raw_signal[t * N_ep : (t + 1) * N_ep]
        filt = bandpass_filter(seg, lo, hi, fs)
        freqs, pxx = signal.welch(filt, fs=fs, nperseg=int(fs * 2),
                                  window="hann", scaling="density")
        alpha_mask = (freqs >= lo) & (freqs <= hi)
        alpha_power[t] = np.trapz(pxx[alpha_mask], freqs[alpha_mask])

    return alpha_power


# --- Build X_t ---
print("Loading EDF signal...")
raw_eeg, sfreq = load_edf_channel(EDF_FILE)
print("Extracting alpha power per epoch...")
X_raw = compute_epoch_alpha_power(raw_eeg, sfreq)
X_t = pd.Series(X_raw, name="alpha_power_uV2")
print(f"Series length: {len(X_t)} epochs  (~{len(X_t) * EPOCH_SEC / 3600:.1f} hours)")


# =============================================================================
# 3. TRAIN / TEST SPLIT  (80 / 20)
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
    axes[0].set_title(f"{title_prefix} Series — Alpha Power (μV²/Hz)")
    axes[0].set_xlabel("Epoch (30 s)")
    axes[0].set_ylabel("Power")

    roll = series.rolling(window=6)
    axes[1].plot(series.values, alpha=0.4, color="#4C72B0", lw=0.7, label="Raw")
    axes[1].plot(roll.mean().values, color="#C44E52", lw=1.5, label="Rolling mean (6 ep)")
    axes[1].plot(roll.std().values,  color="#55A868", lw=1.5, label="Rolling std (6 ep)")
    axes[1].set_title("Rolling Statistics — Trend & Variance Inspection")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=9)

    axes[2].hist(series.dropna(), bins=40, color="#4C72B0", edgecolor="white", alpha=0.8)
    axes[2].set_title("Marginal Distribution of Alpha Power")
    axes[2].set_xlabel("Power (μV²/Hz)")
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
    Variance-stabilising log transform: Y_t = log(X_t + ε)
    ε = 1e-6 * median(X_t) guards against zero-power epochs.
    Appropriate when Var(X_t) is roughly proportional to E[X_t].
    """
    eps = 1e-6 * series.median()
    return np.log(series + eps)


def manual_difference(series: pd.Series, d: int = 1,
                       D: int = 0, S: int = 0) -> pd.Series:
    """
    Apply regular differencing (order d) then seasonal differencing (order D, period S).
    Regular:  ∇^d Y_t  = (1 - B)^d Y_t
    Seasonal: ∇_S^D Y_t = (1 - B^S)^D Y_t
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
    NOTE: ADF only tests for a unit root in the AR part (stochastic trend).
    It does NOT detect other forms of non-stationarity such as structural breaks,
    heteroscedasticity, or deterministic seasonality. Interpret accordingly.
    H0: unit root present (non-stationary). Reject if p < 0.05.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    out = {
        "label"     : label,
        "adf_stat"  : result[0],
        "p_value"   : result[1],
        "n_lags"    : result[2],
        "n_obs"     : result[3],
        "crit_1pct" : result[4]["1%"],
        "crit_5pct" : result[4]["5%"],
        "crit_10pct": result[4]["10%"],
    }
    print(f"\nADF [{label}]:")
    print(f"  Stat       = {out['adf_stat']:.4f}")
    print(f"  p-value    = {out['p_value']:.4f}")
    print(f"  Lags used  = {out['n_lags']}")
    print(f"  Crit 1%    = {out['crit_1pct']:.4f}")
    print(f"  Crit 5%    = {out['crit_5pct']:.4f}")
    print(f"  Crit 10%   = {out['crit_10pct']:.4f}")
    stationary = out["p_value"] < 0.05
    print(f"  → {'STATIONARY' if stationary else 'NON-STATIONARY'} at 5% level")
    return out


# Execute stationarity pipeline
eps_val = 1e-6 * X_t.median()
Y_t = log_transform(train)
run_adf_test(Y_t, "log(X_t)")

W_t = manual_difference(Y_t, d=1)
run_adf_test(W_t, "∇ log(X_t)")

W_t_s = manual_difference(Y_t, d=1, D=1, S=S)
run_adf_test(W_t_s, f"∇∇_{S} log(X_t)")


# =============================================================================
# 6. ACF & PACF PLOTS
# =============================================================================
def plot_acf_pacf(series: pd.Series, lags: int = 60,
                   title: str = "Differenced Series",
                   fname: str = "acf_pacf.png") -> None:
    """
    Side-by-side ACF and PACF with 95% confidence bands.
    Visual identification rules:
      - ACF cuts off at q, PACF tails off → MA(q)
      - PACF cuts off at p, ACF tails off → AR(p)
      - Both tail off → mixed ARMA(p,q)
      - Spikes at multiples of S → seasonal SAR/SMA terms
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(),  lags=lags, alpha=0.05, ax=ax1,
             title=f"ACF — {title}", zero=False)
    plot_pacf(series.dropna(), lags=lags, alpha=0.05, ax=ax2,
              title=f"PACF — {title}", zero=False, method="ywm")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{fname}", bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR}/{fname}")

plot_acf_pacf(W_t_s, lags=60, title=f"∇∇_{S} log(X_t)", fname="acf_pacf.png")

# Also plot the simply differenced series (no seasonal) for comparison
plot_acf_pacf(W_t, lags=60, title="∇ log(X_t)", fname="acf_pacf_nodiff.png")


# =============================================================================
# 7. MANUAL AICc
# =============================================================================
def compute_aicc(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """
    Corrected Akaike Information Criterion for small samples.
        AIC  = -2 * ℓ(θ̂) + 2k
        AICc = AIC + 2k(k+1) / (n - k - 1)
    k = number of free parameters (including σ²), n = effective observations.
    Preferred over AIC when n/k < 40.
    """
    if n_obs - n_params - 1 <= 0:
        raise ValueError(f"Model overparameterized: n={n_obs}, k={n_params}")
    aic  = -2.0 * log_likelihood + 2.0 * n_params
    corr = (2.0 * n_params * (n_params + 1)) / (n_obs - n_params - 1)
    return aic + corr


# =============================================================================
# 8. SARIMA FITTING
# =============================================================================
def fit_sarima(endog: pd.Series, order: tuple, seasonal_order: tuple,
               label: str) -> dict:
    """
    Fit a SARIMAX model and report statistics manually.
    Model: Φ_P(B^S) φ_p(B) ∇^d ∇_S^D Y_t = Θ_Q(B^S) θ_q(B) ε_t
    No automated diagnostics — raw parameter access only.
    """
    model  = SARIMAX(endog, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=True, enforce_invertibility=True,
                     trend="n")
    result = model.fit(disp=False, maxiter=200)

    n_obs    = result.nobs
    n_params = result.df_model + 1   # +1 for σ²
    ll       = result.llf
    aicc_val = compute_aicc(ll, n_params, n_obs)

    print(f"\n{'='*60}")
    print(f"  {label}  SARIMA{order}x{seasonal_order}")
    print(f"  Log-likelihood : {ll:.4f}")
    print(f"  AIC            : {result.aic:.4f}")
    print(f"  AICc           : {aicc_val:.4f}")
    print(f"  BIC            : {result.bic:.4f}")
    print(f"  k (params)     : {n_params}")
    print(f"  n (obs)        : {n_obs}")
    print(f"  Coefficients:")
    for name, val, se in zip(result.param_names,
                              result.params,
                              result.bse):
        print(f"    {name:<20} = {val:+.6f}  (SE={se:.6f})")
    print(f"{'='*60}")

    return {"label": label, "result": result, "aicc": aicc_val,
            "order": order, "seasonal_order": seasonal_order}


model_A = fit_sarima(Y_t, order=(1,1,1), seasonal_order=(1,1,1,S), label="Model A")
model_B = fit_sarima(Y_t, order=(2,1,1), seasonal_order=(0,1,1,S), label="Model B")

best = min([model_A, model_B], key=lambda m: m["aicc"])
print(f"\nBest model by AICc: {best['label']}  (AICc = {best['aicc']:.4f})")


# =============================================================================
# 9. RESIDUAL DIAGNOSTICS
# =============================================================================
def mcleod_li_test(resid: np.ndarray, lags: list = [10, 20, 30]) -> pd.DataFrame:
    """
    McLeod-Li test: Ljung-Box applied to the SQUARED residuals.
    Tests for nonlinear structure / conditional heteroscedasticity (ARCH effects).
    H0: no autocorrelation in squared residuals (linear model is adequate).
    If H0 rejected, a GARCH extension may be warranted.
    """
    squared = pd.Series(resid ** 2)
    lb = acorr_ljungbox(squared, lags=lags, return_df=True)
    print("\nMcLeod-Li Test (Ljung-Box on squared residuals):")
    print(lb.to_string())
    return lb


def plot_residuals(fit_result, label: str = "", fname_prefix: str = "") -> None:
    resid = fit_result.resid.dropna().values

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(resid, color="#4C72B0", lw=0.8)
    ax1.axhline(0, color="red", lw=0.8, ls="--")
    ax1.set_title(f"Residuals — {label}")
    ax1.set_xlabel("Epoch")

    ax2 = fig.add_subplot(gs[1, 0])
    plot_acf(resid, lags=40, alpha=0.05, ax=ax2, zero=False,
             title="ACF of Residuals")

    ax3 = fig.add_subplot(gs[1, 1])
    probplot(resid, dist="norm", plot=ax3)
    ax3.set_title("Q-Q Plot of Residuals")

    plt.tight_layout()
    fname = f"{FIGURES_DIR}/residuals_{fname_prefix}.png"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

    # Ljung-Box on residuals
    lb = acorr_ljungbox(resid, lags=[10, 20, 30], return_df=True)
    print(f"\nLjung-Box test on residuals ({label}):")
    print(lb.to_string())

    # Shapiro-Wilk (first 500 residuals)
    stat, p_sw = shapiro(resid[:500])
    print(f"Shapiro-Wilk: W={stat:.4f}  p={p_sw:.4f}")

    # McLeod-Li test
    mcleod_li_test(resid, lags=[10, 20, 30])


plot_residuals(best["result"], label=best["label"],
               fname_prefix=best["label"].replace(" ", "_"))


# =============================================================================
# 10. FORECASTING & BACK-TRANSFORMATION
# =============================================================================
def forecast_and_backtransform(fit_dict: dict, steps: int,
                                eps: float) -> pd.DataFrame:
    """
    Out-of-sample prediction on log scale, then bias-corrected back-transform.

    Log scale:
        Ŷ_{n+h} = point forecast
        σ²_h    = forecast variance

    Back-transform (log-normal mean):
        X̂_{n+h} = exp(Ŷ_{n+h} + σ²_h / 2) − ε

    95% CI:
        Lower = exp(Ŷ_{n+h} − 1.96·σ_h) − ε
        Upper = exp(Ŷ_{n+h} + 1.96·σ_h) − ε

    The bias correction term σ²_h/2 arises because E[exp(Y)] = exp(μ + σ²/2)
    for Y ~ N(μ, σ²). Omitting it systematically underestimates the mean.
    """
    result  = fit_dict["result"]
    fc      = result.get_forecast(steps=steps)
    fc_mean = fc.predicted_mean
    fc_ci   = fc.conf_int(alpha=0.05)
    fc_var  = fc.var_pred_mean

    df = pd.DataFrame()
    df["log_mean"] = fc_mean.values
    df["log_lo"]   = fc_ci.iloc[:, 0].values
    df["log_hi"]   = fc_ci.iloc[:, 1].values
    df["fc_var"]   = fc_var.values if hasattr(fc_var, "values") else fc_var

    df["forecast"]  = np.exp(df["log_mean"] + df["fc_var"] / 2.0) - eps
    df["lower_95"]  = np.exp(df["log_lo"]) - eps
    df["upper_95"]  = np.exp(df["log_hi"]) - eps

    return df


fc_df = forecast_and_backtransform(best, steps=len(test), eps=eps_val)


def plot_forecast(train_orig: pd.Series, test_orig: pd.Series,
                  fc_df: pd.DataFrame, label: str = "") -> None:
    fig, ax = plt.subplots(figsize=(16, 5))
    n_train = len(train_orig)
    n_test  = len(test_orig)
    t_train = np.arange(n_train)
    t_test  = np.arange(n_train, n_train + n_test)

    ax.plot(t_train, train_orig.values, color="#4C72B0", lw=0.9, label="Training")
    ax.plot(t_test,  test_orig.values,  color="#55A868", lw=0.9, label="Actual (test)")
    ax.plot(t_test,  fc_df["forecast"].values,
            color="#C44E52", lw=1.3, ls="--", label=f"Forecast ({label})")
    ax.fill_between(t_test, fc_df["lower_95"].values, fc_df["upper_95"].values,
                    color="#C44E52", alpha=0.15, label="95% CI")
    ax.set_title(f"Alpha Power Forecast — {label}")
    ax.set_xlabel("Epoch (30 s)")
    ax.set_ylabel("Alpha Power (μV²/Hz)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/forecast.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR}/forecast.png")

    actual = test_orig.values
    pred   = fc_df["forecast"].values[:len(actual)]
    mae    = np.mean(np.abs(actual - pred))
    rmse   = np.sqrt(np.mean((actual - pred) ** 2))
    mape   = np.mean(np.abs((actual - pred) / (actual + 1e-9))) * 100
    print(f"\nForecast Accuracy ({label}):")
    print(f"  MAE  = {mae:.6f}")
    print(f"  RMSE = {rmse:.6f}")
    print(f"  MAPE = {mape:.2f}%")

plot_forecast(train, test, fc_df, label=best["label"])


# =============================================================================
# 11. SPECTRAL ANALYSIS
#     Periodogram, Fisher g-test, KS cumulative periodogram test,
#     and parametric AR PSD
# =============================================================================

def fisher_g_test(Pxx: np.ndarray) -> tuple:
    """
    Fisher's g-test for periodogram significance.
    Tests whether the largest periodogram ordinate is significantly greater
    than expected under the null hypothesis of white noise (iid Gaussian).

    Test statistic:  g = max(I_j) / Σ I_j
    P-value (exact): p = Σ_{k=1}^{floor(1/g)} (-1)^{k-1} C(m,k) (1 - k*g)^{m-1}
    where m = number of Fourier frequencies (excluding zero).

    H0: series is white noise (flat spectrum).
    Reject H0 if p < 0.05 → significant periodicity detected.
    """
    m   = len(Pxx)
    g   = np.max(Pxx) / np.sum(Pxx)
    upper = int(np.floor(1.0 / g))
    p_val = 0.0
    for k in range(1, upper + 1):
        sign = (-1) ** (k - 1)
        p_val += sign * float(comb(m, k, exact=True)) * (1 - k * g) ** (m - 1)
    p_val = np.clip(p_val, 0.0, 1.0)
    print(f"\nFisher g-test:")
    print(f"  g statistic = {g:.6f}")
    print(f"  p-value     = {p_val:.6f}")
    print(f"  → {'Significant periodicity detected' if p_val < 0.05 else 'No significant periodicity'} at 5% level")
    return g, p_val


def ks_cumulative_periodogram(Pxx: np.ndarray) -> tuple:
    """
    Kolmogorov-Smirnov test on the cumulative periodogram.
    Under white noise, the normalized cumulative periodogram
        C(k) = Σ_{j=1}^{k} I_j / Σ_{j=1}^{m} I_j
    should follow a Uniform(0,1) distribution (Bartlett, 1954).

    We apply scipy's two-sided KS test against Uniform(0,1).
    H0: spectral mass is uniformly distributed → white noise.
    Reject H0 if p < 0.05 → residual spectral structure remains.
    """
    m      = len(Pxx)
    cumPxx = np.cumsum(Pxx) / np.sum(Pxx)
    # Theoretical uniform quantiles at the same fractional positions
    uniform_q = np.linspace(1 / m, 1.0, m)
    ks_stat, p_val = kstest(cumPxx, "uniform")
    print(f"\nKS Cumulative Periodogram Test:")
    print(f"  KS statistic = {ks_stat:.6f}")
    print(f"  p-value      = {p_val:.6f}")
    print(f"  → {'Residual spectral structure detected' if p_val < 0.05 else 'No residual structure'} at 5% level")
    return ks_stat, p_val, cumPxx, uniform_q


def spectral_analysis(series: np.ndarray, epoch_sec: int = EPOCH_SEC,
                       ar_max_order: int = 25) -> None:
    """
    Full spectral analysis of the alpha-power series X_t:
      1. Non-parametric periodogram (Schuster)
      2. Fisher g-test for dominant periodicity
      3. KS test on cumulative periodogram (white-noise check)
      4. Parametric AR PSD via Yule-Walker (AutoReg with AIC order selection)

    Frequency axis converted to cycles/hour:
        f_hr = f_epoch * (3600 / epoch_sec)

    Expected peaks:
      ~0.67 cycles/hr (90-min NREM-REM ultradian cycle)
      ~6.67 cycles/hr (9-min spindle-burst cadence)
    """
    n     = len(series)
    dt_hr = epoch_sec / 3600.0     # hours per epoch

    # --- Non-parametric periodogram ---
    f_ep, Pxx = signal.periodogram(series, fs=1.0)   # fs = 1 epoch^{-1}
    f_hr      = f_ep / dt_hr                         # convert to cycles/hour
    # Exclude DC (index 0)
    f_hr_nz  = f_hr[1:]
    Pxx_nz   = Pxx[1:]

    # --- Fisher g-test ---
    g_stat, g_pval = fisher_g_test(Pxx_nz)

    # --- KS cumulative periodogram ---
    ks_stat, ks_pval, cumPxx, unif_q = ks_cumulative_periodogram(Pxx_nz)

    # --- Parametric AR PSD (AutoReg with AIC) ---
    ar_model  = AutoReg(series, lags=ar_max_order, old_names=False).fit()
    # Refit with AIC-selected order
    best_aic  = np.inf
    best_ar   = 1
    for p in range(1, ar_max_order + 1):
        try:
            m = AutoReg(series, lags=p, old_names=False).fit()
            if m.aic < best_aic:
                best_aic = m.aic
                best_ar  = p
        except Exception:
            pass
    ar_fit    = AutoReg(series, lags=best_ar, old_names=False).fit()
    ar_params = ar_fit.params[1:]   # exclude intercept
    sigma2    = np.var(ar_fit.resid)
    print(f"\nAR spectral estimate: order selected by AIC = {best_ar}")

    w    = np.linspace(0, np.pi, 512)
    H    = np.ones(len(w), dtype=complex)
    for k, a_k in enumerate(ar_params, start=1):
        H -= a_k * np.exp(-1j * w * k)
    S_ar = sigma2 / (np.abs(H) ** 2)
    f_ar = (w / (2 * np.pi)) / dt_hr

    # --- Top periodogram peaks ---
    peak_idx = np.argsort(Pxx_nz)[-5:][::-1]
    print("\nTop-5 Periodogram Peaks:")
    for idx in peak_idx:
        cyc_hr     = f_hr_nz[idx]
        period_min = 60.0 / cyc_hr if cyc_hr > 0 else np.inf
        print(f"  f = {cyc_hr:.4f} cyc/hr  →  period ≈ {period_min:.1f} min  "
              f"(Pxx = {Pxx_nz[idx]:.4e})")

    # --- Plots ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Periodogram
    axes[0].semilogy(f_hr_nz, Pxx_nz, color="#4C72B0", lw=0.8)
    axes[0].axvline(1 / 1.5, color="red", lw=1.2, ls="--",
                    label="~90-min NREM-REM cycle (0.67 cyc/hr)")
    axes[0].axvline(1 / 0.15, color="orange", lw=1.0, ls=":",
                    label="~9-min spindle cadence (6.67 cyc/hr)")
    axes[0].set_title("Periodogram of Alpha-Power Series X_t")
    axes[0].set_xlabel("Frequency (cycles / hour)")
    axes[0].set_ylabel("Power (log scale)")
    axes[0].legend(fontsize=9)

    # AR PSD
    axes[1].semilogy(f_ar[1:], S_ar[1:], color="#C44E52", lw=1.0)
    axes[1].axvline(1 / 1.5, color="navy", lw=1.2, ls="--",
                    label=f"~90-min cycle | AR({best_ar}) PSD")
    axes[1].set_title(f"Parametric AR({best_ar}) Spectral Estimate (AIC-selected order)")
    axes[1].set_xlabel("Frequency (cycles / hour)")
    axes[1].set_ylabel("PSD")
    axes[1].legend(fontsize=9)

    # Cumulative periodogram
    freq_positions = np.linspace(0, 1, len(Pxx_nz))
    axes[2].plot(freq_positions, cumPxx, color="#4C72B0", lw=1.2,
                 label="Cumulative periodogram")
    axes[2].plot(freq_positions, freq_positions, color="red", lw=1.0, ls="--",
                 label="Expected under white noise")
    axes[2].set_title(f"Cumulative Periodogram  (KS stat={ks_stat:.4f}, p={ks_pval:.4f})")
    axes[2].set_xlabel("Normalized frequency")
    axes[2].set_ylabel("Cumulative spectral mass")
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/spectral_analysis.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR}/spectral_analysis.png")


spectral_analysis(X_t.values, epoch_sec=EPOCH_SEC)

print("\n=== Pipeline complete. All figures saved to outputs/figures/ ===")
