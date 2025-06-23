import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numpy.linalg import lstsq

# =====================================================================
#  PARAMETERS
# ---------------------------------------------------------------------
#  POLY_DEGREE : degree of the polynomial trend P_d(t)
#  HARMONICS_H : number of Fourier harmonics (denoted by 'h' in plots)
#  EXTRAP_DAYS : days to extrapolate beyond last observation
#  BATCH_SIZE  : how many coordinate points to plot per figure
# =====================================================================
POLY_DEGREE   = 2   # linear trend
HARMONICS_H   = 4   # number of harmonics h
EXTRAP_DAYS   = 365  # forecast horizon (days)
BATCH_SIZE    = 6   # sub‑plots per figure

# =====================================================================
#  DATA LOADING
# =====================================================================
# CSV must contain: date, longitude, latitude, density
# 'date' should be parseable with pandas.to_datetime

df = pd.read_csv('density.csv', parse_dates=['date'])

# unique coordinate pairs (longitude, latitude)
unique_coords = (
    df[['longitude', 'latitude']]
      .drop_duplicates()
      .reset_index(drop=True)
)

# ---------------------------------------------------------------------
#  DESIGN MATRIX CONSTRUCTION
# ---------------------------------------------------------------------

def build_design_matrix(t: np.ndarray,
                        tau: np.ndarray,
                        degree: int,
                        h: int) -> np.ndarray:
    """Create feature matrix X = [poly features | Fourier features].

    Polynomial block: 1, t, t^2, ..., t^degree (captures global trend).
    Fourier block   : cos(2πkτ), sin(2πkτ) for k = 1..h (captures seasonality).
    """
    # Polynomial columns
    poly_cols = [np.ones_like(t)] + [t**p for p in range(1, degree + 1)]

    # Fourier columns (two per harmonic: cos & sin)
    fourier_cols = [func(2 * np.pi * k * tau)
                    for k in range(1, h + 1)
                    for func in (np.cos, np.sin)]

    # Combine and return (shape: M × ((degree+1) + 2h))
    return np.vstack(poly_cols + fourier_cols).T

# ---------------------------------------------------------------------
#  FITTING FUNCTION (OLS)
# ---------------------------------------------------------------------

def fit_trend_fourier(t: np.ndarray,
                      y: np.ndarray,
                      degree: int,
                      h: int):
    """Fit polynomial trend + Fourier seasonality using ordinary least squares.

    Returns:
        coeffs     : fitted coefficients vector
        t0, T      : time normalisation parameters
        predict_fn : vectorised function predicting \hat{y}(t_query)
    """
    # Time normalisation parameters
    t0 = t.min()
    T  = t.max() - t.min()
    tau = (t - t0) / T  # scaled to [0,1]

    # Build design matrix and solve X c = y in least‑squares sense
    X = build_design_matrix(t, tau, degree, h)
    coeffs, *_ = lstsq(X, y, rcond=None)

    # Closure for prediction
    def predict_fn(t_query: np.ndarray):
        tau_q = (t_query - t0) / T
        X_q   = build_design_matrix(t_query, tau_q, degree, h)
        return X_q @ coeffs

    return coeffs, t0, T, predict_fn

# =====================================================================
#  MAIN LOOP OVER COORDINATE POINTS
# =====================================================================
for start in range(0, len(unique_coords), BATCH_SIZE):
    batch = unique_coords.iloc[start:start + BATCH_SIZE]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, batch.iterrows()):
        lon, lat = row['longitude'], row['latitude']

        # ---- select and sort data for current coordinate ----
        sub = (df[(df.longitude == lon) & (df.latitude == lat)]
                 .sort_values('date'))

        # If not enough observations, skip the plot
        if len(sub) < 5:
            ax.text(0.5, 0.5, 'Мало данных', ha='center', va='center')
            ax.set_title(f'Lon: {lon}, Lat: {lat}')
            ax.axis('off')
            continue

        # ------------------------------------------------------
        #  Create time axis (days from first date) and y values
        # ------------------------------------------------------
        dates = sub['date']
        t_days = (dates - dates.iloc[0]).dt.total_seconds() / 86400.0
        y_vals = sub['density'].values

        # Fit model (trend+Fourier)
        coeffs, t0, T, predict = fit_trend_fourier(t_days, y_vals,
                                                   POLY_DEGREE,
                                                   HARMONICS_H)

        # Dense time grid including forward extrapolation horizon
        t_dense = np.linspace(t_days.min(), t_days.max() + EXTRAP_DAYS, 400)
        dates_dense = dates.iloc[0] + pd.to_timedelta(t_dense, unit='D')
        y_dense = predict(t_dense)

        # Training error (Mean Squared Error)
        mse = np.mean((y_vals - predict(t_days))**2)

        # ---------------------------- plotting ---------------------------
        # Raw observations (blue line + markers)
        ax.plot(dates, y_vals, '-o', color='blue', markersize=4,
                label=r'Исходные $\rho(t)$')

        # Trend + Fourier curve (purple)
        ax.plot(dates_dense, y_dense,
                color='purple', label=f'T+F (h={HARMONICS_H})')

        # Title with parameters and compact MSE display
        ax.set_title(
            f'Lon: {lon}, Lat: {lat} | deg={POLY_DEGREE}, h={HARMONICS_H}\n'
            f'MSE = {mse:.4f}',
            fontsize=11
        )

        # Axes labels and cosmetics
        ax.set_xlabel('Дата $t$')
        ax.set_ylabel(r'$\rho$')
        ax.grid(alpha=0.3)

        # Nice date ticks on x‑axis
        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='x', labelrotation=30)
        ax.legend(fontsize=8)

    # Remove empty sub‑plots (if any) in the last batch
    for j in range(len(batch), BATCH_SIZE):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.show()
