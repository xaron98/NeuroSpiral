# CLAUDE.md — Spiral Journey Project Context

## Project Overview
Spiral Journey is a circadian rhythm tracker that maps sleep/wake cycles onto an Archimedean spiral using D3.js + React. It applies chronobiology tools (Cosinor Analysis, Phase Response Curves) to reveal hidden patterns in circadian rhythms.

## Stack
- Vite 6, React 19, D3.js 7, Tailwind CSS 3
- No CRA, no rc-slider, no imperative D3 DOM manipulation
- D3 is used ONLY for math (scales, line generators, interpolation). All rendering is declarative React/JSX SVG.

## Architecture
```
src/
├── App.jsx                    # Main layout & state (viewMode, showCosinor, showPRC, selectedDay, numDays)
├── components/
│   ├── SpiralCanvas.jsx       # Main SVG: spiral path + data points + cosinor overlay + acrophase markers
│   ├── StatsPanel.jsx         # 8 computed metrics from cosinor analysis
│   ├── PRCChart.jsx           # Phase Response Curve with 3 models (light/exercise/melatonin)
│   ├── DriftChart.jsx         # Cumulative circadian drift + amplitude bars
│   ├── DayDetail.jsx          # Selected day: sleep architecture, cosinor params, phase distribution
│   └── Legend.jsx             # Phase colors or viridis scale
├── hooks/
│   └── useSpiralGeometry.js   # Core math: maps (day, hour) → (x, y) on Archimedean spiral r(θ) = a + bθ/2π
└── utils/
    ├── cosinor.js             # Cosinor fit: Y(t) = MESOR + Amp × cos(ω(t - acrophase)). Also: slidingCosinor, acrophaseDrift, rhythmStability
    ├── phaseResponse.js       # PRC models: lightPRC, exercisePRC, melatoninPRC + generatePRCCurve
    └── sleepData.js           # generateSleepData (demo), formatHour, formatDate, calculateStats, SLEEP_PHASES
```

## Scientific Background

### Cosinor Analysis
- Fits cosine curve to 24h activity data
- Extracts: MESOR (baseline), Amplitude (rhythm strength), Acrophase (peak time), Period, R²
- Sliding window (7-day) version implemented but not yet wired to UI
- Acrophase drift tracks cumulative phase shift day-over-day

### Phase Response Curve (PRC)
- Same stimulus has different effect depending on circadian time
- 3 zones: dead zone (midday), advance zone (morning), delay zone (evening)
- 3 models: bright light, exercise, melatonin (opposite to light)

### Spiral Representation
- Archimedean spiral: each revolution = 1 day
- Points at same angle across revolutions = same time of day on different dates
- Enables visual detection of periodic patterns and phase drift
- Cosinor curve overlaid as offset from spiral centerline using normal vectors

### Key References
- Weber et al. (2001) — Visualizing Time-Series on Spirals
- Gu et al. (2021) — spiralize R package, Bioinformatics
- Cornelissen (2014) — Cosinor-based rhythmometry
- Khalsa et al. (2003) — PRC to bright light pulses
- Penttonen & Buzsáki (2003) — logarithmic relationship between brain oscillators

## Design
- Dark theme: bg #0c0e14, accent #5bffa8 (green), surfaces #12151e
- Sleep phase colors: deep=#1a1a6e, REM=#6e3fa0, light=#5b8bd4, awake=#f5c842
- Fonts: JetBrains Mono (code/data), Space Grotesk (display)
- Tailwind custom theme in tailwind.config.js with `spiral-*` color tokens
- Custom CSS components: .panel, .stat-card, .btn, .btn-active, .btn-inactive

## Current State & Known Issues
- Data is demo-generated (generateSleepData). No real data import yet.
- slidingCosinor() exists in utils but isn't connected to UI
- No data persistence / localStorage
- No export functionality

## Planned Next Steps (priority order)
1. CSV import for real sleep data (Sleep Cycle, Fitbit, Apple Health formats)
2. Wire up slidingCosinor to show 7-day rolling cosinor parameters
3. Event markers on spiral (coffee, exercise, light exposure) with PRC-predicted shift
4. Manual sleep logging input
5. Correlate points across spiral revolutions (same angular position = same time of day)
6. Export analysis as PDF report
7. Logarithmic spiral option (for modeling circadian drift with non-24h period)

## Code Conventions
- Functional components with hooks only
- useMemo for expensive calculations (spiral geometry, data mapping, cosinor)
- useCallback for event handlers
- No useEffect for D3 DOM manipulation — everything is declarative JSX SVG
- Tailwind classes directly, custom classes only in index.css @layer components
- ES module imports, no CommonJS
