# Datasets

All datasets used in this study are publicly available. No dataset is included in this repository — download from the original sources.

## Sleep Polysomnography

### HMC (primary dataset)
- **Source:** https://physionet.org/content/hmc-sleep-staging/
- **Size:** 151 subjects, ~117,510 epochs
- **Channels:** EEG (C4-M1), ECG, EOG (E1-M2), EMG (chin)
- **Sampling rate:** 200/500 Hz (resampled to 100 Hz)
- **Labels:** AASM (W, N1, N2, N3, REM)
- **Access:** PhysioNet credentialed access

### CAP Sleep Database
- **Source:** https://physionet.org/content/capslpdb/
- **Size:** 16 healthy + 18 RBD subjects
- **Channels:** EEG, ECG, EOG, EMG
- **Sampling rate:** 512 Hz
- **Labels:** AASM
- **Access:** Open

### Sleep-EDF Expanded
- **Source:** https://physionet.org/content/sleep-edfx/
- **Size:** 78 recordings, 76 subjects (ages 25-101)
- **Channels:** EEG (Fpz-Cz, Pz-Oz), EOG, EMG (no ECG)
- **Sampling rate:** 100 Hz
- **Labels:** R&K → AASM mapped
- **Access:** Open

### DREAMT
- **Source:** https://physionet.org/content/dreamt/
- **Size:** 100 subjects with wearable + PSG
- **Channels:** PSG (12-channel) + wearable (BVP, ACC, HR)
- **Sampling rates:** 100 Hz (PSG), 64 Hz (wearable)
- **Labels:** AASM
- **Access:** Open

## Cardiology

### PTB-XL
- **Source:** https://physionet.org/content/ptb-xl/
- **Size:** 21,837 ECG recordings, 12-lead, 10 seconds
- **Sampling rate:** 500 Hz
- **Labels:** 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP)
- **Access:** Open

## Climate

### Meteostat (Barcelona Aeropuerto)
- **Source:** Meteostat Python library (`pip install meteostat`)
- **Station:** 08181 (Barcelona / Aeropuerto)
- **Period:** 2024 (full year, hourly)
- **Variables:** temp, humidity, pressure, wind speed, dew point

## Finance

### Yahoo Finance
- **Source:** `pip install yfinance`
- **Tickers:** AAPL (2020-2024)
- **Variables:** close, volume, volatility, RSI, MACD
