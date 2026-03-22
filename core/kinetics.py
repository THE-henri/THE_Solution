import numpy as np
from scipy.optimize import curve_fit


def exp_decay(t, A0, k):
    return A0 * np.exp(-k * t)


def exp_negative(t, A0, A_inf, k):
    return A_inf + (A0 - A_inf) * np.exp(-k * t)


def fit_half_life(time, absorbance, switch="positive", A_inf_manual=None):

    time = np.array(time)
    absorbance = np.array(absorbance)

    if len(time) < 3:
        print("Not enough data points for fitting.")
        return None, None, None

    try:

        # -------------------------
        # POSITIVE SWITCH
        # -------------------------
        if switch == "positive":

            p0 = [absorbance[0], 1e-3]
            popt, _ = curve_fit(exp_decay, time, absorbance, p0=p0)

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
                popt_temp, _ = curve_fit(model_fixed, time, absorbance, p0=p0)

                A0, k = popt_temp
                popt = (A0, A_inf_manual, k)
                fitted_curve = model_fixed(time, A0, k)

            else:
                p0 = [absorbance[0], absorbance[-1], 1e-3]
                popt, _ = curve_fit(exp_negative, time, absorbance, p0=p0)

                A0, A_inf, k = popt
                fitted_curve = exp_negative(time, A0, A_inf, k)

        else:
            raise ValueError("switch must be 'positive' or 'negative'")

        k = popt[-1]
        t_half = np.log(2) / k

        # R² calculation
        ss_res = np.sum((absorbance - fitted_curve)**2)
        ss_tot = np.sum((absorbance - np.mean(absorbance))**2)
        r_squared = 1 - ss_res/ss_tot

        return popt, t_half, fitted_curve, r_squared

    except Exception as e:
        print(f"Error fitting half-life: {e}")
        return None, None, None, None
