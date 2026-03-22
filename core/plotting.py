import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt


def plot_extinction_coefficients(df_result, show=True):
    """
    Plot extinction coefficients with error bars per preparation and overall.

    Each preparation is drawn as a solid line with ±1 std error bars.
    The overall mean across all preparations is drawn in black on top.

    Parameters
    ----------
    df_result : pandas.DataFrame
        DataFrame returned by `calculate_extinction_coefficients_integer_wavelengths`.
        Expected columns: "Wavelength (nm)", Prep1_Mean, Prep1_Std, …, Mean, Std
    show : bool
        If True, immediately shows the plot. Can be False for GUI integration.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    wavelengths = df_result["Wavelength (nm)"].values
    compound    = df_result["Compound"].iloc[0] if "Compound" in df_result.columns else ""
    title       = f"Extinction Coefficients — {compound}" if compound else "Extinction Coefficients"

    fig, ax = plt.subplots(figsize=(9, 5))

    # Per-preparation: solid line + error bars (±1 std from replicate scans)
    prep_cols = [col for col in df_result.columns if "_Mean" in col and "Prep" in col]
    for col in prep_cols:
        prep_name = col.replace("_Mean", "")
        std_col   = prep_name + "_Std"
        mean_vals = df_result[col].values
        std_vals  = df_result[std_col].values if std_col in df_result.columns else None

        ax.errorbar(
            wavelengths, mean_vals,
            yerr=std_vals,
            fmt="-",
            linewidth=1.2,
            elinewidth=0.5,
            capsize=0,
            label=prep_name,
        )

    # Overall mean ± std across preparations (inter-prep variability)
    if "Mean" in df_result.columns and "Std" in df_result.columns:
        ax.errorbar(
            wavelengths, df_result["Mean"].values,
            yerr=df_result["Std"].values,
            fmt="-",
            color="black",
            linewidth=2,
            elinewidth=0.8,
            capsize=0,
            label="Overall mean ± std",
        )

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"$\varepsilon$ (M$^{-1}$ cm$^{-1}$)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    if show:
        plt.show()

    return fig, ax



def plot_half_life(time, absorbance, start_idx=None, end_idx=None, time_sel=None, absorbance_sel=None, highlight_color="orange",fitted_curve=None, r_squared=None, title=None, show=True):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(time, absorbance, 'o', markeredgecolor='grey', markerfacecolor='none', markersize=4, label='Data')

    if start_idx is not None and end_idx is not None:
        ax.plot(time[start_idx:end_idx+1], absorbance[start_idx:end_idx+1],
                markeredgecolor=highlight_color,markerfacecolor='none', marker='o', linestyle='', markersize=4, label='Selected')

    if fitted_curve is not None:
        ax.plot(time_sel, fitted_curve, "--", color="red",markerfacecolor='none', linewidth=2, label="Fit")

    if r_squared is not None:
        ax.text(0.05, 0.95, f"$R^2$ = {r_squared:.6f}",
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Absorbance")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True)

    if show:
        plt.show()

    return fig, ax


def plot_half_life_with_linear(time, absorbance, start_idx=None, end_idx=None,
                                time_sel=None, absorbance_sel=None,
                                time_outliers=None, absorbance_outliers=None,
                                fitted_curve=None, r_squared=None,
                                popt=None, t_half=None, switch="negative",
                                title=None, show=True):
    """
    Two-panel plot: left = exponential fit, right = linearised ln(A - A∞) vs time.

    For a negative switch: plots ln(A - A∞) vs time (should be linear if first-order).
    For a positive switch: plots ln(A) vs time.
    A linear fit is overlaid on the right panel.
    Outlier points (if provided) are shown in red on both panels.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")

    # --- Left panel: exponential fit ---
    ax1.plot(time, absorbance, 'o', markeredgecolor='grey', markerfacecolor='none', markersize=4, label='Data')
    if start_idx is not None and end_idx is not None:
        ax1.plot(time[start_idx:end_idx + 1], absorbance[start_idx:end_idx + 1],
                 markeredgecolor='orange', markerfacecolor='none', marker='o',
                 linestyle='', markersize=4, label='Selected')
    if time_outliers is not None and len(time_outliers) > 0:
        ax1.plot(time_outliers, absorbance_outliers, 'x', color='red',
                 markersize=6, markeredgewidth=1.5, label='Outliers')
    if fitted_curve is not None:
        ax1.plot(time_sel, fitted_curve, "--", color="red", linewidth=2, label="Exp. fit")
    if r_squared is not None:
        ax1.text(0.05, 0.95, f"$R^2$ = {r_squared:.6f}",
                 transform=ax1.transAxes, fontsize=12, verticalalignment='top')

    if popt is not None:
        if switch == "negative":
            A0, A_inf_p, k = popt[0], popt[1], popt[2]
            eq_line   = r"$A(t)=A_\infty+(A_0-A_\infty)\,e^{-kt}$"
            param_lines = (f"$A_0$ = {A0:.4f}\n"
                           f"$A_\\infty$ = {A_inf_p:.4f}\n"
                           f"$k$ = {k:.4f} s$^{{-1}}$")
        else:
            A0, k = popt[0], popt[1]
            eq_line   = r"$A(t)=A_0\,e^{-kt}$"
            param_lines = (f"$A_0$ = {A0:.4f}\n"
                           f"$k$ = {k:.4f} s$^{{-1}}$")

        t_half_line = f"$t_{{1/2}}$ = {t_half:.2f} s" if t_half is not None else ""
        annotation  = eq_line + "\n" + param_lines
        if t_half_line:
            annotation += "\n" + t_half_line

        ax1.text(0.97, 0.97, annotation,
                 transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Absorbance")
    if title:
        ax1.set_title(title)
    ax1.legend()
    ax1.grid(True)

    # --- Right panel: linearised ---
    if popt is not None and time_sel is not None and absorbance_sel is not None:
        if switch == "negative":
            A_inf = popt[1]
            delta_A = absorbance_sel - A_inf
            valid = np.abs(delta_A) > 1e-10
            y_label = "ln|Absorbance \u2212 A\u221e|"
        else:
            delta_A = absorbance_sel
            valid = delta_A > 0
            y_label = "ln(Absorbance)"

        if valid.sum() > 1:
            t_valid = time_sel[valid]
            ln_vals = np.log(np.abs(delta_A[valid]))
            ax2.plot(t_valid, ln_vals, 'o', markeredgecolor='orange',
                     markerfacecolor='none', markersize=4, label='Data (linearised)')
            coeffs = np.polyfit(t_valid, ln_vals, 1)
            k_linear = -coeffs[0]
            ax2.plot(t_valid, np.polyval(coeffs, t_valid), '--', color='red',
                     linewidth=2, label=f"Linear fit  k = {k_linear:.4f} s\u207b\u00b9")

        # Overlay outliers on the linearised panel
        if time_outliers is not None and len(time_outliers) > 0:
            if switch == "negative":
                delta_out = absorbance_outliers - popt[1]
                loggable = delta_out != 0
            else:
                delta_out = absorbance_outliers
                loggable = delta_out > 0
            if loggable.sum() > 0:
                ax2.plot(time_outliers[loggable], np.log(np.abs(delta_out[loggable])),
                         'x', color='red', markersize=6, markeredgewidth=1.5,
                         label='Outliers')

        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel(y_label)
        if title:
            ax2.set_title(f"{title} \u2014 Linearised")
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, (ax1, ax2)
