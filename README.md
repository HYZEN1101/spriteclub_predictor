# SpriteClub Predictor

A machine learning prediction system for [SpriteClub](https://spriteclub.tv) MUGEN betting. Uses a 4-model ensemble trained on 1.5 million historical matches to predict who wins before you place your bet.
Please download the model pkl file from the releases and place it into the main folder to make the project work as intended

---

## Versions

| | [Desktop App](./DESKTOP_APP.md) | [Web App](./WEB_APP.md) |
|---|---|---|
| **Interface** | Tkinter window | Browser (localhost) |
| **Input** | Paste/type manually | **Fully automatic** |
| **Twitch integration** | ❌ | ✅ Auto-detects matches |
| **Auto-logging** | ❌ Click to log | ✅ Bot announces → logged |
| **Portable EXE** | ✅ PyInstaller | ❌ |
| **Recommended for** | Quick use, offline | Live betting sessions |

---

## Model

### Architecture

Four models run in parallel, blended with per-format weights optimised on held-out recent data:

```
XGBoost  (per format)  ─┐
LightGBM (per format)  ─┤─ weighted blend → prediction
RandomForest (per fmt) ─┤
RandomForest (global)  ─┘
```

### Formats supported

- `OVO` — 1v1
- `TVT` — 2v2 simultaneous
- `TVO` — 1 vs team (or team vs 1)
- `Turns2` / `Turns3` — tag/turns format
- `XVX` — variable size

### Accuracy on recent matches

| Format | Accuracy |
|--------|----------|
| OVO 1v1 | ~83% |
| TVO mixed | ~88% |
| Turns | ~85–87% |
| TVT 2v2 | ~74% |

### Training data

- ~1.5 million matches from 2016–2025
- Time-weighted with 12-month half-life (recent matches weighted up to 1.5×)
- 19,667 characters in the lookup database
- 1.18 million unique head-to-head pairs tracked

### Features

| Feature | Notes |
|---------|-------|
| Elo | Rating at match time |
| Winstreak | Current streak (capped ±30) |
| Tier | Div1–Div6 + New/Untiered |
| LifeMax / PowerMax | Character base stats |
| Attack / Defense | Combat stats |
| Crashes | Reliability signal |
| Head-to-head rate | Win rate between these two specifically |
| Team aggregates | Avg, min, max across team for multi-char formats |

### Online learning

Every logged outcome is stored as a correction sample. Every 20 corrections, an XGBoost adapter is retrained in a background thread and takes priority for that format. The model continuously improves from your session data.

---

## Quick Start

### Desktop

```bash
pip install scikit-learn xgboost lightgbm customtkinter numpy
python app.py
```

### Web

```bash
pip install flask flask-socketio numpy scikit-learn xgboost lightgbm
python server.py
# open http://localhost:5000
```

---

## Confidence Guide

| Label | Win probability | Action |
|-------|----------------|--------|
| VERY HIGH | 82%+ | Strong bet |
| HIGH | 72%+ | Good bet |
| MODERATE | 62%+ | Lean this way |
| SLIGHT | 54%+ | Small bet |
| COIN FLIP | <54% | Skip |

---

## Notes

- Characters added to SpriteClub after the dataset snapshot will show as `NOT FOUND`
- Fuzzy name matching handles typos — flagged in amber so you can verify
- The web app connects to Twitch IRC anonymously — no account or token required
- Dataset is from early 2025; scraping fresh data would further improve accuracy

---

## Docs

- [Desktop App →](./DESKTOP_APP.md)
- [Web App →](./WEB_APP.md)
