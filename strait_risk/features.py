"""Stage 1: seasonal baseline per series, daily anomalies, weekly features.

All features are log-ratios against the strait's own normal traffic, so they
are unitless and comparable across straits: -0.69 means half of normal,
-2.3 means a tenth of normal.
"""

import numpy as np
import pandas as pd

# Calm periods used to fit "normal" traffic: before COVID, and after the COVID
# recovery but before the Panama drought (mid-2023) and the Red Sea crisis.
CALM = [("2019-01-01", "2020-02-29"), ("2021-07-01", "2023-05-31")]
# HMM inputs. "size" and "vol" are still computed and shown, but left out of the
# model: in the variant runs, vol turned the worst state into "typhoon week"
# (Taiwan Strait, Bosporus), and size spent a state on above-normal weeks.
FEATURES = ["level", "gap"]
LEVEL_CAP = 0.3  # log-ratio, about 35% above normal


def calm_mask(index):
    m = np.zeros(len(index), bool)
    for a, b in CALM:
        m |= (index >= a) & (index <= b)
    return m


def _design(index):
    """Intercept + 3 yearly Fourier harmonics + day-of-week dummies (no trend)."""
    doy = index.dayofyear.to_numpy() / 365.25
    cols = [np.ones(len(index))]
    for k in (1, 2, 3):
        cols += [np.sin(2 * np.pi * k * doy), np.cos(2 * np.pi * k * doy)]
    dow = index.dayofweek.to_numpy()
    cols += [(dow == d).astype(float) for d in range(1, 7)]
    return np.column_stack(cols)


def expected(index, y, calm):
    """Fit the baseline on calm days, robustly: drop days with residuals > 4 robust
    standard deviations and refit, so stray AIS outages do not bend "normal"."""
    X = _design(index)
    use = calm & np.isfinite(y)
    for _ in range(3):
        beta, *_ = np.linalg.lstsq(X[use], y[use], rcond=None)
        resid = y - X @ beta
        mad = 1.4826 * np.nanmedian(np.abs(resid[use]))
        use = calm & np.isfinite(y) & (np.abs(resid) < 4 * mad)
    return X @ beta


def daily(df):
    """Daily log-anomalies (observed vs expected) per series."""
    calm = calm_mask(df.index)
    out = pd.DataFrame(index=df.index)
    for col in ("n_total", "n_container", "n_tanker"):
        y = np.log(df[col].to_numpy(float) + 0.5)
        out[col] = y - expected(df.index, y, calm)
        out[col + "_expected"] = np.exp(expected(df.index, y, calm)) - 0.5
    n = df["n_total"].to_numpy(float)
    size = np.where(n > 0, np.log((df["capacity"].to_numpy(float) + 1) / np.maximum(n, 1)), np.nan)
    out["size_expected"] = expected(df.index, size, calm)
    out["capacity_obs"] = df["capacity"].to_numpy(float)
    out["n_total_obs"] = n
    out["n_container_obs"] = df["n_container"].to_numpy(float)
    out["n_tanker_obs"] = df["n_tanker"].to_numpy(float)
    out.attrs["data_through"] = df.attrs.get("data_through")
    return out


def weekly(day):
    """Weekly feature table (weeks ending Sunday, at least 5 valid days)."""
    g = day.resample("W-SUN")
    w = pd.DataFrame({
        "level": g["n_total"].mean(),           # all transits vs normal
        "container": g["n_container"].mean(),
        "gap": g["n_tanker"].mean() - g["n_container"].mean(),  # >0: tankers hold up better
        # Average ship size over the whole week (total deadweight / total transits),
        # vs expected. Pooling the week keeps it stable when daily counts are tiny.
        "size": np.log((g["capacity_obs"].sum() + 1) / np.maximum(g["n_total_obs"].sum(), 1))
                - g["size_expected"].mean(),
        "days": g["n_total"].count(),
    })
    # Excess volatility: day-to-day spread of total traffic beyond what pure
    # counting noise (Var log n ~ 1/n) would give at that traffic level.
    var = g["n_total"].var()
    noise = g["n_total_obs"].apply(lambda s: np.mean(1 / (s.dropna() + 0.5)) if s.notna().any() else np.nan)
    w["vol"] = np.sqrt(np.maximum(0, var - noise))
    w["obs_total"] = g["n_total_obs"].mean()
    w["exp_total"] = g["n_total_expected"].mean()
    w["obs_container"] = g["n_container_obs"].mean()
    w["exp_container"] = g["n_container_expected"].mean()
    w["obs_tanker"] = g["n_tanker_obs"].mean()
    w["exp_tanker"] = g["n_tanker_expected"].mean()
    w = w[w["days"] >= 5].drop(columns="days").dropna(subset=FEATURES)
    # Rule-based Stage 1 index: share of expected total traffic that is missing.
    # Its three bands also seed the HMM's three states.
    w["D"] = 1 - np.exp(w["level"])
    # Above-normal traffic is not a risk signal. Without a cap, a surge (the Cape
    # of Good Hope taking Red Sea diversions, +75%) fits the wide Closed state
    # better than the narrow Normal one. Cap the model input at about +35%.
    w["level"] = w["level"].clip(upper=LEVEL_CAP)
    w["rule_state"] = pd.cut(w["D"], [-np.inf, 0.15, 0.40, np.inf],
                             labels=["Normal", "Reduced", "Closed"]).astype(str)
    w.attrs["data_through"] = day.attrs.get("data_through")
    return w
