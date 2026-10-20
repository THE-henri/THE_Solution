import warnings
import numpy as np
from scipy.optimize import curve_fit


def exp_decay(t, A0, k):
    return A0 * np.exp(-k * t)


def exp_negative(t, A0, A_inf, k):
    return A_inf + (A0 - A_inf) * np.exp(-k * t)


def _compute_linear_k(time, absorbance, switch, A_inf_manual):
    """
    Compute k (and A0) by linear regression on ln-transformed data (always available).

    Positive switch, A_inf = 0:   ln(A) = ln(A0) - k*t
    Positive switch, A_inf ≠ 0:   ln|A - A_inf| = ln|A0 - A_inf| - k*t
    Negative switch:               ln|A - A_inf| = ln|A0 - A_inf| - k*t

    Time is shifted so t=0 is the first point of `time` (i.e. the start of the
    analysed window/segment) before fitting, so the recovered A0 is the
    amplitude AT that window's start rather than an extrapolation back to
    absolute t=0 — the latter overflows to nonsense magnitudes when the
    window starts far from t=0 (e.g. a segment beginning at t=2000 s).

    Returns (k_linear, A0_linear, t_half_linear, r2_linear), or
    (None, None, None, None) if it fails. R² is computed on the linearised
    (ln-transformed) data.
    """
    try:
        A_inf  = float(A_inf_manual) if A_inf_manual is not None else None
        t_offset = float(time[0])
        t_shifted = time - t_offset

        if switch == "positive" and (A_inf is None or A_inf == 0.0):
            # Pure decay to zero: ln(A) is linear in t
            A_inf = A_inf if A_inf is not None else 0.0
            valid = absorbance > 0
            if valid.sum() < 2:
                return None, None, None, None
            y = np.log(absorbance[valid])
            coeffs = np.polyfit(t_shifted[valid], y, 1)
            k  = float(-coeffs[0])
            A0 = float(np.exp(coeffs[1]))
        else:
            # Negative switch (any A_inf) or positive with non-zero baseline
            if A_inf is None:
                A_inf = float(absorbance[-1])
            delta = absorbance - A_inf
            valid = np.abs(delta) > 1e-12
            if valid.sum() < 2:
                return None, None, None, None
            y = np.log(np.abs(delta[valid]))
            coeffs = np.polyfit(t_shifted[valid], y, 1)
            k  = float(-coeffs[0])
            # Sign recovered from the first valid point, since ln|.| loses it.
            A0 = float(A_inf + np.sign(delta[valid][0]) * np.exp(coeffs[1]))

        if k <= 0:
            return None, None, None, None

        y_hat  = np.polyval(coeffs, t_shifted[valid])
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        return k, A0, float(np.log(2) / k), r2

    except Exception:
        return None, None, None, None


def _linear_fallback(time, absorbance, switch, A_inf_manual):
    """
    Fallback when curve_fit fails: use linear regression on ln-transformed data.

    Returns (popt, t_half, fitted_curve, r_squared, k_linear, t_half_linear,
             r2_linear, A0_linear).
    popt is always 3-element (A0, A_inf, k) for both switch types.
    """
    try:
        A_inf_val = float(A_inf_manual) if A_inf_manual is not None else None

        # Shift time to start at 0 so that exp(intercept) equals the amplitude
        # at the first data point rather than an extrapolation to absolute t=0.
        # This prevents overflow when the fitting window starts at large t values.
        t_offset = float(time[0])

        if switch == "positive":
            if A_inf_val is not None and A_inf_val != 0.0:
                # Fixed non-zero baseline
                delta = absorbance - A_inf_val
                valid = np.abs(delta) > 1e-12
                if valid.sum() < 2:
                    return None, None, None, None, None, None, None, None
                t_v = time[valid] - t_offset
                y_v = np.log(np.abs(delta[valid]))
                coeffs = np.polyfit(t_v, y_v, 1)
                k  = float(-coeffs[0])
                A0 = float(A_inf_val + np.sign(delta[valid][0]) * np.exp(coeffs[1]))
                if k <= 0:
                    k = 1e-9
                fitted_curve = A_inf_val + (A0 - A_inf_val) * np.exp(-k * (time - t_offset))
                popt = (A0, A_inf_val, k)
            else:
                # A_inf = 0 (default or explicit zero)
                A_inf_val = A_inf_val if A_inf_val is not None else 0.0
                valid = absorbance > 0
                if valid.sum() < 2:
                    return None, None, None, None, None, None, None, None
                t_v = time[valid] - t_offset
                y_v = np.log(absorbance[valid])
                coeffs = np.polyfit(t_v, y_v, 1)
                k  = float(-coeffs[0])
                A0 = float(np.exp(coeffs[1]))
                if k <= 0:
                    k = 1e-9
                fitted_curve = A0 * np.exp(-k * (time - t_offset))
                popt = (A0, A_inf_val, k)

        else:  # negative
            A_inf = A_inf_val if A_inf_val is not None else float(absorbance[-1])
            delta = absorbance - A_inf
            valid = np.abs(delta) > 1e-12
            if valid.sum() < 2:
                return None, None, None, None, None, None, None, None
            t_v = time[valid] - t_offset
            y_v = np.log(np.abs(delta[valid]))
            coeffs = np.polyfit(t_v, y_v, 1)
            k  = float(-coeffs[0])
            A0 = float(A_inf + np.sign(delta[valid][0]) * np.exp(coeffs[1]))
            if k <= 0:
                k = 1e-9
            if A_inf_manual is not None:
                fitted_curve = A_inf_val + (A0 - A_inf_val) * np.exp(-k * (time - t_offset))
                popt = (A0, A_inf_val, k)
            else:
                fitted_curve = A_inf + (A0 - A_inf) * np.exp(-k * (time - t_offset))
                popt = (A0, A_inf, k)

        t_half = np.log(2) / k
        ss_res = np.sum((absorbance - fitted_curve) ** 2)
        ss_tot = np.sum((absorbance - np.mean(absorbance)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        y_hat     = np.polyval(coeffs, t_v)
        ls_res    = float(np.sum((y_v - y_hat) ** 2))
        ls_tot    = float(np.sum((y_v - np.mean(y_v)) ** 2))
        r2_linear = float(1 - ls_res / ls_tot) if ls_tot > 0 else float("nan")
        return popt, t_half, fitted_curve, r_squared, k, t_half, r2_linear, A0

    except Exception as e:
        print(f"  Linear fallback also failed: {e}")
        return None, None, None, None, None, None, None, None


def fit_half_life(time, absorbance, switch="positive", A_inf_manual=None):
    """
    Fit a first-order decay or build-up curve.

    Returns
    -------
    (popt, t_half, fitted_curve, r_squared, k_linear, t_half_linear,
     r2_linear, A0_linear)

    popt is always 3-element: (A0, A_inf, k) for both switch types. A0 is the
    amplitude at the start of this window (time[0]), not at absolute t=0 —
    see the t_offset note below.
      - Positive, A_inf free:  A_inf extracted from data by 3-param fit.
      - Positive, A_inf fixed: A_inf = A_inf_manual (may be 0 for pure decay).
      - Negative, A_inf free:  A_inf extracted from data by 3-param fit.
      - Negative, A_inf fixed: A_inf = A_inf_manual.

    k_linear / A0_linear / t_half_linear / r2_linear — from ln-transform
    linear regression. None when A_inf is free (unknown until after the
    exponential fit).
    """
    time       = np.array(time)
    absorbance = np.array(absorbance)

    if len(time) < 2:
        print("Not enough data points for fitting.")
        return None, None, None, None, None, None, None, None

    # Shift time so t=0 is this window's first point. A0 then means "the
    # amplitude at the start of the analysed window", not an extrapolation
    # back to the experiment's absolute t=0 — the latter overflows to
    # nonsense magnitudes (e.g. 1e30) once the window starts far from t=0,
    # since it requires exp(+k * time[0]).
    t_offset  = float(time[0])
    t_shifted = time - t_offset

    # k_linear can only be computed when A_inf is already known (fixed).
    if A_inf_manual is not None:
        k_linear, A0_linear, t_half_linear, r2_linear = _compute_linear_k(
            time, absorbance, switch, A_inf_manual)
    else:
        k_linear, A0_linear, t_half_linear, r2_linear = None, None, None, None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            # ── POSITIVE SWITCH ──────────────────────────────────────────────
            if switch == "positive":

                if A_inf_manual is not None:
                    # Fixed baseline — may be 0 (pure decay) or non-zero (offset)
                    A_inf_fixed = float(A_inf_manual)

                    def model_fixed_pos(t, A0, k):
                        return A_inf_fixed + (A0 - A_inf_fixed) * np.exp(-k * t)

                    p0     = [absorbance[0], 1e-3]
                    bounds = ([0, 0], [np.inf, np.inf])
                    popt_temp, _ = curve_fit(model_fixed_pos, t_shifted, absorbance,
                                             p0=p0, bounds=bounds, maxfev=2000)
                    A0, k = popt_temp
                    popt         = (A0, A_inf_fixed, k)
                    fitted_curve = model_fixed_pos(t_shifted, A0, k)

                else:
                    # Free baseline — extract A_inf from the decay itself
                    p0     = [absorbance[0], absorbance[-1], 1e-3]
                    bounds = ([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf])
                    popt, _ = curve_fit(exp_negative, t_shifted, absorbance,
                                        p0=p0, bounds=bounds, maxfev=2000)
                    A0, A_inf, k = popt
                    fitted_curve = exp_negative(t_shifted, A0, A_inf, k)
                    # A_inf is now known — compute linear k
                    k_linear, A0_linear, t_half_linear, r2_linear = _compute_linear_k(
                        time, absorbance, switch, A_inf)

            # ── NEGATIVE SWITCH ──────────────────────────────────────────────
            elif switch == "negative":

                if A_inf_manual is not None:

                    def model_fixed(t, A0, k):
                        return A_inf_manual + (A0 - A_inf_manual) * np.exp(-k * t)

                    p0     = [absorbance[0], 1e-3]
                    bounds = ([-np.inf, 0], [np.inf, np.inf])
                    popt_temp, _ = curve_fit(model_fixed, t_shifted, absorbance,
                                             p0=p0, bounds=bounds, maxfev=2000)
                    A0, k = popt_temp
                    popt         = (A0, A_inf_manual, k)
                    fitted_curve = model_fixed(t_shifted, A0, k)

                else:
                    p0     = [absorbance[0], absorbance[-1], 1e-3]
                    bounds = ([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf])
                    popt, _ = curve_fit(exp_negative, t_shifted, absorbance,
                                        p0=p0, bounds=bounds, maxfev=2000)
                    A0, A_inf, k = popt
                    fitted_curve = exp_negative(t_shifted, A0, A_inf, k)
                    # A_inf now known — compute linear k
                    k_linear, A0_linear, t_half_linear, r2_linear = _compute_linear_k(
                        time, absorbance, switch, A_inf)

            else:
                raise ValueError("switch must be 'positive' or 'negative'")

        k      = popt[-1]
        t_half = np.log(2) / k

        ss_res    = np.sum((absorbance - fitted_curve) ** 2)
        ss_tot    = np.sum((absorbance - np.mean(absorbance)) ** 2)
        r_squared = 1 - ss_res / ss_tot

        return (popt, t_half, fitted_curve, r_squared,
                k_linear, t_half_linear, r2_linear, A0_linear)

    except Exception as e:
        print(f"  curve_fit failed ({e})")
        # Fall back to linear regression when A_inf is known or for positive switch
        # (positive falls back assuming A_inf = 0 if not specified).
        if A_inf_manual is not None or switch == "positive":
            print("  Falling back to linear regression on ln-transform.")
            return _linear_fallback(time, absorbance, switch, A_inf_manual)
        else:
            print("  Cannot fall back: A∞ unknown without a successful exponential fit.")
            return None, None, None, None, None, None, None, None
