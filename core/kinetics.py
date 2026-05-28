import warnings
import numpy as np
from scipy.optimize import curve_fit


def exp_decay(t, A0, k):
    return A0 * np.exp(-k * t)


def exp_negative(t, A0, A_inf, k):
    return A_inf + (A0 - A_inf) * np.exp(-k * t)


def _compute_linear_k(time, absorbance, switch, A_inf_manual):
    """
    Compute k by linear regression on ln-transformed data (always available).

    For positive switch:   ln(A) = ln(A0) - k*t
    For negative switch:   ln|A - A_inf| = ln|A0 - A_inf| - k*t

    Returns (k_linear, t_half_linear, r2_linear) or (None, None, None) if it fails.
    R² is computed on the linearised (ln-transformed) data.
    """
    try:
        if switch == "positive":
            valid = absorbance > 0
            if valid.sum() < 2:
                return None, None, None
            y = np.log(absorbance[valid])
            coeffs = np.polyfit(time[valid], y, 1)
            k = float(-coeffs[0])
        else:
            A_inf = float(A_inf_manual) if A_inf_manual is not None \
                    else float(absorbance[-1])
            delta = absorbance - A_inf
            valid = np.abs(delta) > 1e-12
            if valid.sum() < 2:
                return None, None, None
            y = np.log(np.abs(delta[valid]))
            coeffs = np.polyfit(time[valid], y, 1)
            k = float(-coeffs[0])

        if k <= 0:
            return None, None, None

        y_hat  = np.polyval(coeffs, time[valid])
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        return k, float(np.log(2) / k), r2

    except Exception:
        return None, None, None


def _linear_fallback(time, absorbance, switch, A_inf_manual):
    """
    Fallback when curve_fit fails: use linear regression on ln-transformed data.

    Returns (popt, t_half, fitted_curve, r_squared, k_linear, t_half_linear, r2_linear).
    """
    try:
        if switch == "positive":
            valid = absorbance > 0
            if valid.sum() < 2:
                return None, None, None, None, None, None, None
            t_v = time[valid]
            y_v = np.log(absorbance[valid])
            coeffs = np.polyfit(t_v, y_v, 1)
            k  = float(-coeffs[0])
            A0 = float(np.exp(coeffs[1]))
            if k <= 0:
                k = 1e-9
            fitted_curve = exp_decay(time, A0, k)
            popt = (A0, k)

        else:  # negative
            A_inf = float(A_inf_manual) if A_inf_manual is not None \
                    else float(absorbance[-1])
            delta = absorbance - A_inf
            valid = np.abs(delta) > 1e-12
            if valid.sum() < 2:
                return None, None, None, None, None, None, None
            t_v = time[valid]
            y_v = np.log(np.abs(delta[valid]))
            coeffs = np.polyfit(t_v, y_v, 1)
            k  = float(-coeffs[0])
            A0 = float(A_inf + np.sign(delta[valid][0]) * np.exp(coeffs[1]))
            if k <= 0:
                k = 1e-9
            if A_inf_manual is not None:
                fitted_curve = A_inf_manual + (A0 - A_inf_manual) * np.exp(-k * time)
                popt = (A0, A_inf_manual, k)
            else:
                fitted_curve = exp_negative(time, A0, A_inf, k)
                popt = (A0, A_inf, k)

        t_half = np.log(2) / k
        ss_res = np.sum((absorbance - fitted_curve) ** 2)
        ss_tot = np.sum((absorbance - np.mean(absorbance)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        # R² on ln-transformed data (linear fit quality)
        y_hat  = np.polyval(coeffs, t_v)
        ls_res = float(np.sum((y_v - y_hat) ** 2))
        ls_tot = float(np.sum((y_v - np.mean(y_v)) ** 2))
        r2_linear = float(1 - ls_res / ls_tot) if ls_tot > 0 else float("nan")
        # When curve_fit failed, linear k IS the primary k
        return popt, t_half, fitted_curve, r_squared, k, t_half, r2_linear

    except Exception as e:
        print(f"  Linear fallback also failed: {e}")
        return None, None, None, None, None, None


def fit_half_life(time, absorbance, switch="positive", A_inf_manual=None):
    """
    Fit a first-order decay or build-up curve.

    Returns
    -------
    (popt, t_half, fitted_curve, r_squared, k_linear, t_half_linear, r2_linear)

    popt / t_half / fitted_curve / r_squared  — from exponential fit (curve_fit),
        or from the linear fallback if curve_fit failed.
    k_linear / t_half_linear / r2_linear  — always from the ln-transform linear
        regression, regardless of whether curve_fit succeeded.
        None if not computable (negative+free before curve_fit succeeds).
    """
    time = np.array(time)
    absorbance = np.array(absorbance)

    if len(time) < 2:
        print("Not enough data points for fitting.")
        return None, None, None, None, None, None, None

    # Linear k is only meaningful when A∞ is known.
    # For negative+free we defer until after curve_fit provides the fitted A∞.
    if switch == "positive" or A_inf_manual is not None:
        k_linear, t_half_linear, r2_linear = _compute_linear_k(
            time, absorbance, switch, A_inf_manual)
    else:
        k_linear, t_half_linear, r2_linear = None, None, None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            # -------------------------
            # POSITIVE SWITCH
            # -------------------------
            if switch == "positive":

                p0 = [absorbance[0], 1e-3]
                bounds = ([0, 0], [np.inf, np.inf])
                popt, _ = curve_fit(exp_decay, time, absorbance,
                                    p0=p0, bounds=bounds, maxfev=2000)

                A0, k = popt
                fitted_curve = exp_decay(time, A0, k)

            # -------------------------
            # NEGATIVE SWITCH
            # -------------------------
            elif switch == "negative":

                if A_inf_manual is not None:

                    def model_fixed(t, A0, k):
                        return A_inf_manual + (A0 - A_inf_manual) * np.exp(-k * t)

                    p0 = [absorbance[0], 1e-3]
                    bounds = ([-np.inf, 0], [np.inf, np.inf])
                    popt_temp, _ = curve_fit(model_fixed, time, absorbance,
                                             p0=p0, bounds=bounds, maxfev=2000)

                    A0, k = popt_temp
                    popt = (A0, A_inf_manual, k)
                    fitted_curve = model_fixed(time, A0, k)

                else:
                    p0 = [absorbance[0], absorbance[-1], 1e-3]
                    bounds = ([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf])
                    popt, _ = curve_fit(exp_negative, time, absorbance,
                                        p0=p0, bounds=bounds, maxfev=2000)

                    A0, A_inf, k = popt
                    fitted_curve = exp_negative(time, A0, A_inf, k)

                    # Now A∞ is known from the fit — compute linear k with it.
                    k_linear, t_half_linear, r2_linear = _compute_linear_k(
                        time, absorbance, switch, A_inf)

            else:
                raise ValueError("switch must be 'positive' or 'negative'")

        k = popt[-1]
        t_half = np.log(2) / k

        # R² calculation
        ss_res = np.sum((absorbance - fitted_curve)**2)
        ss_tot = np.sum((absorbance - np.mean(absorbance))**2)
        r_squared = 1 - ss_res/ss_tot

        return popt, t_half, fitted_curve, r_squared, k_linear, t_half_linear, r2_linear

    except Exception as e:
        print(f"  curve_fit failed ({e})")
        # Linear fallback is only valid when A∞ is known (positive switch or fixed A∞).
        # For negative+free we cannot determine A∞ without the exponential fit.
        if switch == "positive" or A_inf_manual is not None:
            print("  Falling back to linear regression on ln-transform.")
            return _linear_fallback(time, absorbance, switch, A_inf_manual)
        else:
            print("  Cannot fall back: A∞ is unknown without a successful exponential fit.")
            return None, None, None, None, None, None, None
