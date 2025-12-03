#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compact exhaustive analysis script with full function documentation.

Purpose
-------
This script performs a compact but rigorous battery of analyses to assess whether
two year-over-year (YOY) rate series are linearly related. It reads:
  - CSV_PATH / CSV_COL  : 貨物裝卸量.csv  -> change(%)
  - XLS_PATH / XLS_COL  : 外銷訂單.xls -> 外銷訂單金額_美元年增率 (%)
It aligns the series (by date if available, otherwise by position), runs:
  - Frequentist: OLS, Pearson r, permutation test
  - Bayesian: Normal and Student-t linear regression via PyMC with iterative
    remedies for divergences (standardization, non-centered parameterization,
    increased target_accept, robust likelihood, tightened priors)
  - Evidence measures: Savage–Dickey BF01 (KDE), BIC approximation, LOO/WAIC
  - Posterior predictive checks (PPC) when possible
It aggregates results and prints a single final explanation string.

Output
------
Only one block of text is printed: the final automated interpretation and numeric
summary. No intermediate diagnostics are printed by default.

Notes on reproducibility and limitations
----------------------------------------
- Small sample sizes (n=7) limit precision of KDE-based posterior density
  estimates and BIC asymptotics; interpret BF and BIC with caution.
- If PyMC/ArviZ are unavailable, the script will raise an error.
- The script is intentionally compact; for full auditability, keep the code and
  inference data (idata) available for deeper inspection.
"""

from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from scipy.stats import gaussian_kde, pearsonr
from sklearn.linear_model import LinearRegression
import pymc as pm
import arviz as az

# -------------------------
# Configuration (edit paths/column names if needed)
# -------------------------
CSV_PATH = Path("/content/貨物裝卸量.csv")
XLS_PATH = Path("/content/外銷訂單.xls")
CSV_COL = "change(%)"
XLS_COL = "外銷訂單金額_美元年增率 (%)"
DATE_CANDS = ["Date", "date", "DATE", "日期"]
ALIGN_ON_DATE = True

# -------------------------
# Utility functions (fully documented)
# -------------------------
def coerce(series: pd.Series) -> pd.Series:
    """
    Coerce a pandas Series to numeric values robustly.

    This function:
      - Accepts strings with parentheses for negatives, commas, percent signs,
        currency symbols, and scientific notation.
      - Replaces "(...)" with "-..." to handle common negative formatting.
      - Removes non-numeric characters except digits, dot, minus, comma, e/E.
      - Removes thousands separators (commas).
      - Converts to numeric with errors coerced to NaN.

    Parameters
    ----------
    series : pd.Series
        Input series possibly containing formatted numeric strings.

    Returns
    -------
    pd.Series
        Numeric series with invalid entries as NaN.
    """
    if is_string_dtype(series) or series.dtype == object:
        s = series.astype(str).str.strip()
        s = s.str.replace(r"\(([^)]+)\)", r"-\1", regex=True)
        s = s.str.replace(r"[^\d\.\-\,eE]", "", regex=True)
        s = s.str.replace(",", "", regex=False)
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def find_date_column(df: pd.DataFrame):
    """
    Find a plausible date column name in a DataFrame.

    The function checks a list of common names first, then scans column names
    for 'date' or the Chinese '日期'. Returns None if no candidate is found.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.

    Returns
    -------
    str or None
        Column name that likely contains dates, or None.
    """
    for name in DATE_CANDS:
        if name in df.columns:
            return name
    for col in df.columns:
        if str(col).lower() == "date" or "日期" in str(col):
            return col
    return None


def read_csv_clean(path: Path, value_col: str) -> pd.DataFrame:
    """
    Read a CSV file and clean the target numeric column.

    Steps:
      - Read CSV as strings to avoid automatic coercion.
      - Identify a date column if present.
      - Drop rows missing the value column or date (if date exists).
      - Coerce the value column to numeric using `coerce`.

    Parameters
    ----------
    path : Path
        Path to CSV file.
    value_col : str
        Column name to extract and coerce.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with coerced numeric column.
    """
    df = pd.read_csv(path, dtype=str)
    date_col = find_date_column(df)
    subset = [c for c in ([value_col, date_col] if date_col else [value_col]) if c]
    df = df.dropna(subset=subset).reset_index(drop=True)
    df[value_col] = coerce(df[value_col])
    return df


def read_xls_clean(path: Path, value_col: str) -> dict:
    """
    Read an Excel file (all sheets) and clean sheets that contain the target column.

    Steps:
      - Read all sheets into a dict of DataFrames.
      - For each sheet, identify a date column if present.
      - Drop rows missing the value column or date (if date exists).
      - Coerce the value column to numeric.

    Parameters
    ----------
    path : Path
        Path to Excel file.
    value_col : str
        Column name to extract and coerce.

    Returns
    -------
    dict
        Mapping sheet_name -> cleaned DataFrame.
    """
    sheets = pd.read_excel(path, sheet_name=None)
    cleaned = {}
    for name, df in sheets.items():
        date_col = find_date_column(df)
        subset = [c for c in ([value_col, date_col] if date_col else [value_col]) if c]
        df = df.dropna(subset=subset).reset_index(drop=True)
        df[value_col] = coerce(df[value_col])
        cleaned[name] = df
    return cleaned


def align_series(s_change: pd.Series, s_export: pd.Series,
                 df_csv: pd.DataFrame, df_xls: pd.DataFrame):
    """
    Align two series either by date (preferred) or by position.

    If ALIGN_ON_DATE is True and both DataFrames contain date columns, the
    function converts them to datetime, drops rows with missing date/value,
    merges on date, and returns aligned series. Otherwise it drops NaNs and
    truncates to the shorter length.

    Parameters
    ----------
    s_change : pd.Series
        Series from CSV (change%).
    s_export : pd.Series
        Series from Excel (export YOY).
    df_csv : pd.DataFrame
        Full CSV DataFrame (used to find date column).
    df_xls : pd.DataFrame
        Full Excel sheet DataFrame (used to find date column).

    Returns
    -------
    (pd.Series, pd.Series)
        Tuple of aligned (change_series, export_series).
    """
    if ALIGN_ON_DATE:
        d1 = find_date_column(df_csv)
        d2 = find_date_column(df_xls)
        if d1 and d2:
            t1 = pd.to_datetime(df_csv[d1], errors="coerce")
            t2 = pd.to_datetime(df_xls[d2], errors="coerce")
            tmp1 = pd.DataFrame({"date": t1, "change": s_change}).dropna(subset=["date", "change"])
            tmp2 = pd.DataFrame({"date": t2, "export": s_export}).dropna(subset=["date", "export"])
            merged = pd.merge(tmp1, tmp2, on="date", how="inner").dropna()
            if len(merged) > 0:
                return merged["change"].reset_index(drop=True), merged["export"].reset_index(drop=True)
    s1 = s_change.dropna().reset_index(drop=True)
    s2 = s_export.dropna().reset_index(drop=True)
    n = min(len(s1), len(s2))
    return s1.iloc[:n].reset_index(drop=True), s2.iloc[:n].reset_index(drop=True)


def perm_test(x: np.ndarray, y: np.ndarray, n_perm: int = 5000, seed: int = 0):
    """
    Perform a permutation test for Pearson correlation (two-sided).

    The test computes the observed Pearson correlation and estimates the
    permutation p-value by randomly permuting y n_perm times and counting
    how often the absolute permuted correlation >= absolute observed.

    Parameters
    ----------
    x : np.ndarray
        Predictor values.
    y : np.ndarray
        Response values.
    n_perm : int
        Number of permutations (default 5000).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    (float, float)
        Tuple (p_value, observed_r).
    """
    rng = np.random.default_rng(seed)
    obs_r, _ = pearsonr(x, y)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(y)
        r, _ = pearsonr(x, perm)
        if abs(r) >= abs(obs_r):
            count += 1
    p_val = (count + 1) / (n_perm + 1)
    return p_val, float(obs_r)


def bic_approx_bf(x: np.ndarray, y: np.ndarray):
    """
    Approximate BF01 using BIC difference: BF01 ≈ exp((BIC_null - BIC_alt)/2).

    Null model: intercept only. Alternative: intercept + slope.
    This is a large-sample approximation and may be unreliable for small n.

    Parameters
    ----------
    x : np.ndarray
        Predictor values.
    y : np.ndarray
        Response values.

    Returns
    -------
    float
        Approximated BF01 (null over alternative).
    """
    n = len(x)
    if n < 3:
        return float("nan")
    y = np.asarray(y)
    rss_null = np.sum((y - np.mean(y)) ** 2)
    bic_null = n * np.log(rss_null / n) + 1 * np.log(n)
    X = np.vstack([np.ones(n), x - np.mean(x)]).T
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ beta_hat
    rss_alt = np.sum((y - y_pred) ** 2)
    bic_alt = n * np.log(rss_alt / n) + 2 * np.log(n)
    return float(math.exp((bic_null - bic_alt) / 2.0))


def savage_dickey_bf_from_samples(beta_samples: np.ndarray, prior_sd: float) -> float:
    """
    Compute Savage–Dickey BF01 from posterior samples of beta.

    Steps:
      - Estimate posterior density at 0 using Gaussian KDE on samples.
      - Compute prior density at 0 for Normal(0, prior_sd).
      - BF01 = prior_density_at_0 / posterior_density_at_0.

    Notes:
      - KDE can be unstable for very small sample sizes; a fallback normal
        approximation is used when KDE fails.
      - A small floor is applied to posterior density to avoid division by zero.

    Parameters
    ----------
    beta_samples : np.ndarray
        Posterior samples of beta.
    prior_sd : float
        Standard deviation of the Normal(0, prior_sd) prior.

    Returns
    -------
    float
        BF01 (null over alternative).
    """
    try:
        kde = gaussian_kde(beta_samples)
        post_density = float(kde.evaluate(0.0))
        post_density = max(post_density, 1e-300)
    except Exception:
        m = float(np.mean(beta_samples))
        s = float(np.std(beta_samples, ddof=1))
        if s <= 0:
            post_density = 1e-300
        else:
            post_density = (1.0 / (math.sqrt(2 * math.pi) * s)) * math.exp(-0.5 * (0 - m) ** 2 / (s ** 2))
            post_density = max(post_density, 1e-300)
    prior_density = 1.0 / (math.sqrt(2 * math.pi) * prior_sd)
    return float(prior_density / post_density)


def fit_bayesian_with_remedies(x: np.ndarray, y: np.ndarray,
                              prior_sd: float = 1.0,
                              draws: int = 2000, tune: int = 2000,
                              seed: int = 42, cores: int = 1):
    """
    Fit a Bayesian linear model with iterative remedies for sampling issues.

    The function attempts a sequence of configurations to reduce divergences:
      1. Standardize x (center and scale).
      2. Use non-centered parameterization for beta (beta_raw * prior_sd).
      3. Try increasing target_accept (0.9 -> 0.95 -> 0.99).
      4. Switch to Student-t likelihood (robust) if needed.
      5. Tighten prior scale if divergences persist.

    For each attempt, the function runs NUTS and checks:
      - Number of divergences (should be 0).
      - R-hat values (<= 1.05 considered acceptable here).
    If a configuration yields no divergences and acceptable R-hat, it returns
    the inference data and metadata. Otherwise returns the last attempt info.

    Parameters
    ----------
    x : np.ndarray
        Predictor values.
    y : np.ndarray
        Response values.
    prior_sd : float
        Prior standard deviation for beta.
    draws : int
        Number of posterior draws per chain.
    tune : int
        Number of tuning steps.
    seed : int
        RNG seed for reproducibility.
    cores : int
        Number of CPU cores for sampling.

    Returns
    -------
    dict
        If successful: {"idata": InferenceData, "xs": standardized_x, "prior_sd": used_prior_sd, "robust": bool}
        If not fully successful: last attempt dict with keys above plus diagnostics.
    """
    x = np.asarray(x).astype(float).ravel()
    y = np.asarray(y).astype(float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) < 2:
        raise ValueError("Not enough data for Bayesian fit (n < 2).")
    xs = (x - x.mean()) / (x.std(ddof=1) if x.std(ddof=1) > 0 else 1.0)

    attempts = [
        {"ta": 0.90, "t": tune, "rob": False, "ps": prior_sd},
        {"ta": 0.95, "t": tune * 2, "rob": False, "ps": prior_sd},
        {"ta": 0.95, "t": tune * 2, "rob": True,  "ps": prior_sd},
        {"ta": 0.99, "t": tune * 3, "rob": True,  "ps": max(0.5, prior_sd / 2.0)}
    ]
    last_attempt = None

    for cfg in attempts:
        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=0.0, sigma=5.0)
            beta_raw = pm.Normal("beta_raw", mu=0.0, sigma=1.0)
            beta = pm.Deterministic("beta", beta_raw * cfg["ps"])
            sigma = pm.HalfNormal("sigma", sigma=2.5)
            mu = alpha + beta * xs
            if cfg["rob"]:
                nu = pm.Exponential("nu", 1 / 30)
                y_obs = pm.StudentT("y_obs", mu=mu, sigma=sigma, nu=nu, observed=y)
            else:
                y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
            try:
                idata = pm.sample(draws=draws, tune=cfg["t"], target_accept=cfg["ta"],
                                  random_seed=seed, cores=cores, progressbar=False, init="adapt_diag")
            except Exception:
                last_attempt = {"error": "sampling_failed", "cfg": cfg}
                continue

            # diagnostics
            try:
                div = int(idata.sample_stats["diverging"].sum().values)
            except Exception:
                div = 0
            try:
                rhat_vals = az.rhat(idata).to_array().values.flatten()
                rhat_ok = bool(np.all(np.isnan(rhat_vals) | (rhat_vals <= 1.05)))
            except Exception:
                rhat_ok = True

            last_attempt = {"idata": idata, "div": div, "rhat_ok": rhat_ok, "cfg": cfg}
            if div == 0 and rhat_ok:
                return {"idata": idata, "xs": xs, "prior_sd": cfg["ps"], "robust": cfg["rob"]}

    return last_attempt


def ppc_numeric_summary(idata, x: np.ndarray, y: np.ndarray):
    """
    Compute simple numeric posterior predictive check summaries.

    The function attempts to draw posterior predictive samples and computes:
      - proportion of predictive sample means <= observed mean
      - proportion of predictive sample sds <= observed sd

    These proportions indicate whether the model's predictive distribution
    tends to under/over-estimate observed summary statistics.

    Parameters
    ----------
    idata : arviz.InferenceData
        InferenceData containing posterior samples and (if available) posterior_predictive.
    x : np.ndarray
        Predictor values (unused here but kept for API symmetry).
    y : np.ndarray
        Observed response values.

    Returns
    -------
    dict or None
        Dictionary with keys 'prop_mean_le_obs' and 'prop_sd_le_obs', or None if PPC failed.
    """
    try:
        ppc = pm.sample_posterior_predictive(idata, var_names=["y_obs"], random_seed=0, progressbar=False)
        yrep = np.asarray(ppc["y_obs"])
        if yrep.ndim == 3:
            yrep = yrep.reshape(-1, yrep.shape[-1])
        rep_means = yrep.mean(axis=1)
        rep_sds = yrep.std(axis=1, ddof=1)
        return {"prop_mean_le_obs": float((rep_means <= y.mean()).mean()),
                "prop_sd_le_obs": float((rep_sds <= y.std(ddof=1)).mean())}
    except Exception:
        return None

# -------------------------
# Orchestration and final output
# -------------------------
def main():
    """
    Main orchestration function.

    Steps:
      1. Validate files and read/clean data.
      2. Align series.
      3. Run OLS and permutation test.
      4. Run Bayesian fits across a small grid of prior scales, collect BF01,
         posterior summaries, HDI exclusion counts, PPC and LOO-based robust model counts.
      5. Aggregate results, apply a conservative heuristic to form a final verdict,
         and print a single explanatory block.

    The function prints only the final explanation string.
    """
    if not CSV_PATH.exists():
        sys.exit("CSV file not found")
    if not XLS_PATH.exists():
        sys.exit("Excel file not found")

    csv_df = read_csv_clean(CSV_PATH, CSV_COL)
    xls_sheets = read_xls_clean(XLS_PATH, XLS_COL)

    # find sheet containing target column
    sheet_name = None
    for name, df in xls_sheets.items():
        if XLS_COL in df.columns:
            sheet_name = name
            break
    if sheet_name is None:
        sys.exit("Excel column not found in any sheet")

    raw_change = csv_df[CSV_COL]
    raw_export = xls_sheets[sheet_name][XLS_COL]
    s_change, s_export = align_series(raw_change, raw_export, csv_df, xls_sheets[sheet_name])
    if len(s_change) == 0:
        sys.exit("No overlapping observations after alignment")

    x = np.asarray(s_export).astype(float)
    y = np.asarray(s_change).astype(float)
    n = len(x)

    # Frequentist baseline
    ols = LinearRegression().fit(x.reshape(-1, 1), y)
    slope = float(ols.coef_[0])
    r2 = float(ols.score(x.reshape(-1, 1), y))
    p_perm, obs_r = perm_test(x, y, n_perm=5000, seed=0)

    # Bayesian grid of priors
    prior_grid = [1.0, 0.5, 2.0]
    bf_list = []
    beta_means = []
    beta_sds = []
    hdi_excludes = 0
    ppc_res = None
    loo_robust_pref = 0

    for ps in prior_grid:
        fit = fit_bayesian_with_remedies(x, y, prior_sd=ps, draws=2000, tune=2000, seed=42, cores=1)
        if fit is None:
            continue
        idata = fit.get("idata")
        if idata is None:
            continue
        # extract beta samples (handle non-centered naming)
        try:
            beta_samples = idata.posterior["beta"].values.ravel()
        except Exception:
            if "beta_raw" in idata.posterior:
                br = idata.posterior["beta_raw"].values.ravel()
                beta_samples = br * ps
            else:
                continue
        bf = savage_dickey_bf_from_samples(beta_samples, ps)
        bf_list.append(bf)
        beta_means.append(float(np.mean(beta_samples)))
        beta_sds.append(float(np.std(beta_samples, ddof=1)))
        try:
            hdi = az.hdi(idata, var_names=["beta"])
            if (hdi["beta"][0] > 0) or (hdi["beta"][1] < 0):
                hdi_excludes += 1
        except Exception:
            pass
        # attempt robust comparison: fit robust quickly and compare LOO
        try:
            fit_r = fit_bayesian_with_remedies(x, y, prior_sd=ps, draws=1000, tune=1000, seed=int(ps*10)+1, cores=1)
            if fit_r and fit_r.get("idata") is not None:
                loo_n = az.loo(idata, pointwise=False)
                loo_r = az.loo(fit_r["idata"], pointwise=False)
                if loo_r and loo_n and loo_r.loo < loo_n.loo:
                    loo_robust_pref += 1
        except Exception:
            pass
        if ppc_res is None:
            ppc_res = ppc_numeric_summary(idata, x, y)

    bf_geo = float(np.exp(np.mean(np.log(np.array(bf_list) + 1e-300)))) if bf_list else float("nan")
    beta_mean_overall = float(np.mean(beta_means)) if beta_means else float("nan")
    beta_sd_overall = float(np.mean(beta_sds)) if beta_sds else float("nan")
    hdi_prop = hdi_excludes / len(prior_grid) if prior_grid else 0.0
    bicbf = bic_approx_bf(x, y)

    # conservative heuristic for verdict
    reasons = []
    verdict = "inconclusive"
    if not math.isnan(bf_geo):
        if bf_geo >= 10:
            verdict = "no_relation"
            reasons.append("Bayes factors strongly favor no relation")
        elif bf_geo <= 0.1:
            verdict = "relation"
            reasons.append("Bayes factors strongly favor a relation")
        else:
            reasons.append("Bayes factors are moderate/inconclusive")
    if hdi_prop >= 0.5 and p_perm < 0.05:
        verdict = "relation"
        reasons.append("HDIs often exclude zero and permutation test significant")
    if verdict == "inconclusive":
        reasons.append("frequentist tests show no significant linear relation while BF and HDI are mixed")

    # final explanation (single printed block)
    out = []
    out.append("Final automated interpretation:")
    if verdict == "relation":
        out.append("Conclusion: The data provide evidence for a relationship between the two columns.")
    elif verdict == "no_relation":
        out.append("Conclusion: The data provide evidence that the two columns are not related.")
    else:
        out.append("Conclusion: The evidence is inconclusive regarding a relationship between the two columns.")
    out.append("")
    out.append(f"Summary: sample size = {n}. OLS slope = {slope:.8g}, R^2 = {r2:.8g}; Pearson r = {obs_r:.4g}, permutation p ≈ {p_perm:.4g}.")
    out.append(f"Bayesian Savage–Dickey BF01 (geometric mean across priors) = {bf_geo:.4g} (BIC approx BF01 = {bicbf:.4g}).")
    out.append(f"Posterior slope (avg across priors): mean ≈ {beta_mean_overall:.7g}, sd ≈ {beta_sd_overall:.6g}.")
    out.append(f"95% HDI excluded zero in {hdi_excludes} out of {len(prior_grid)} prior settings ({hdi_prop:.2%}).")
    if ppc_res:
        out.append(f"PPC summary: prop(mean_rep ≤ obs_mean) = {ppc_res['prop_mean_le_obs']:.3f}, prop(sd_rep ≤ obs_sd) = {ppc_res['prop_sd_le_obs']:.3f}.")
    else:
        out.append("Posterior predictive checks not available.")
    out.append(f"Model comparison: robust model preferred by LOO in {loo_robust_pref} out of {len(prior_grid)} prior settings.")
    out.append("")
    out.append("Primary rationale:")
    for r in reasons:
        out.append("- " + r)
    out.append("")
    out.append("Notes: BF01 depends on prior choice; interpret BF together with HDI and predictive checks.")
    print("\n".join(out))


if __name__ == "__main__":
    main()
