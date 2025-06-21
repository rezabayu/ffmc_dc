# src/data/ffmc_calc.py
import numpy as np
import pandas as pd

def _ffmc_one_step(t, rh, w, r, ff_prev):
    """
    One-day FFMC update (Van Wagner 1987).
    Inputs
    -------
    t  : air temperature (°C, local 13 LT is fine)
    rh : relative humidity (%)
    w  : mean wind speed (km h-1)
    r  : 24-h rain (mm, already adjusted so it ‘falls’ on this day)
    ff_prev : yesterday’s FFMC

    Returns
    -------
    ff_today : new FFMC value
    """

    # 1. previous day's mo
    po = (147.2 * (101.0 - ff_prev)) / (59.5 + ff_prev)

    # 2. rainfall adjustment
    if r > 0.5:
        rf = r - 0.5
        mo = po + 42.5 * rf * np.exp(-100.0 / (251.0 - po)) \
                 * (1.0 - np.exp(-6.93 / rf)) \
             + 0.0015 * (po - 150.0)**2 * np.sqrt(rf)
        if mo > 250.:
            mo = 250.
    else:
        mo = po

    # 3. drying / wetting calculation
    if rh > 100.:
        rh = 100.
    elif rh < 0.:
        rh = 0.

    ed = 0.942 * rh**0.679 + \
         (11.0 * np.exp((rh - 100.0) / 10.0)) + \
         (0.18 * (21.1 - t) * (1.0 - np.exp(-0.115 * rh)))

    if mo < ed:
        kl = 0.424 * (1.0 - (rh / 100.0)**1.7) + \
             (0.0694 * np.sqrt(w) * (1.0 - (rh / 100.0)**8))
        kw = kl * 0.581 * np.exp(0.0365 * t)
        m  = ed - (ed - mo) * 10.0**(-kw)
    else:
        ew = 0.618 * rh**0.753 + \
             (10.0 * np.exp((rh - 100.0) / 10.0)) + \
             (0.18 * (21.1 - t) * (1.0 - np.exp(-0.115 * rh)))
        kl = 0.424 * (1.0 - ((100.0 - rh) / 100.0)**1.7) + \
             (0.0694 * np.sqrt(w) * (1.0 - ((100.0 - rh) / 100.0)**8))
        kw = kl * 0.581 * np.exp(0.0365 * t)
        m  = ew + (mo - ew) * 10.0**(-kw)

    # 4. back-transform to FFMC
    ffmc = (59.5 * (250.0 - m)) / (147.2 + m)
    return max(min(ffmc, 101.0), 0.0)


def ffmc_series(df, init_ffmc=85.0):
    """
    Compute a full FFMC time-series for a *single station* DataFrame
    sorted by date.
    Expects columns:
        'temp_13LT_C', 'rh_avg_pc', 'wind_avg_kmh', 'rain_mm'

    Returns
    -------
    pandas.Series of FFMC values
    """
    ff = np.empty(len(df), dtype=float)
    ff_prev = init_ffmc
    for i, (_, row) in enumerate(df.iterrows()):
        ff_today = _ffmc_one_step(row.temp_13LT_C,
                                  row.rh_avg_pc,
                                  row.wind_avg_kmh,
                                  row.rain_mm,
                                  ff_prev)
        ff[i] = ff_today
        ff_prev = ff_today
    return pd.Series(ff, index=df.index, name="ffmc")
