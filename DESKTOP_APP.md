# SpriteClub Predictor — Desktop App

A machine learning prediction tool for [SpriteClub](https://spriteclub.tv) MUGEN betting. Paste a match announcement, get an instant prediction on who to bet on.

---

## Features

- **4-model ensemble** — XGBoost + LightGBM + RandomForest (per-format) + RandomForest (global), with optimised per-format weights
- **All match formats** — OVO (1v1), TVT (2v2), TVO (mixed), Turns2, Turns3
- **Character stats panel** — Tier, Elo, Winstreak, W/L, HP/Attack/Defense, crash warnings
- **Fuzzy name matching** — handles typos and near-matches, flagged in amber
- **Head-to-head history** — shows historical record between the two characters (1v1)
- **Online learning** — logs your outcome feedback and retrains every 20 corrections
- **Session tracker** — tracks bets, profit, accuracy across a session; saves to `sessions.json`
- **Dark UI** — built with CustomTkinter, Segoe UI font

---

## Accuracy

Trained on ~1.5 million historical matches (2016–2025), time-weighted so recent matches matter more.

| Format | Accuracy (recent matches) |
|--------|--------------------------|
| OVO 1v1 | ~83% |
| TVO mixed | ~88% |
| Turns | ~85–87% |
| TVT 2v2 | ~74% |

---

## Installation

**Requirements:** Python 3.9+

```bash
pip install scikit-learn xgboost lightgbm customtkinter numpy
```

**Run:**

```bash
python app.py
```

---

## Usage

1. Paste the SpriteClub chat line into the input box:
   ```
   a [ Char1 ⇒ Char2 ] Vs. b [ Char3 ⇒ Char4 ]
   ```
   or type a simple 1v1:
   ```
   Ryu vs Ken
   ```
2. Press **Predict** or hit `Enter`
3. After the match, click the outcome button with the winner's name
4. The model learns from the correction in the background

### Session Tracking

- Click **Start Session** and enter your starting balance
- Set your bet amount in the bottom field
- Log outcomes after each match
- Click **End Session** for a full report — saved to `sessions.json`

---

## File Structure

```
app.py                ← main application
models_v4.pkl         ← XGBoost + LightGBM per format
models_per_type.pkl   ← RandomForest per format
model_v3.pkl          ← global RF fallback
ensemble_weights.pkl  ← per-format blending weights
char_by_name.pkl      ← character name lookup
char_by_id.pkl        ← character ID lookup
h2h.pkl               ← head-to-head history
sessions.json         ← your bet history (auto-created)
online_data.npz       ← correction samples (auto-created)
models_adapted.pkl    ← adapted models (auto-created)
```

---

## Building an EXE

```bash
pip install pyinstaller

pyinstaller --onedir --windowed --name "SpriteClub Predictor" \
  --add-data "models_v4.pkl;." \
  --add-data "models_per_type.pkl;." \
  --add-data "model_v3.pkl;." \
  --add-data "ensemble_weights.pkl;." \
  --add-data "char_by_name.pkl;." \
  --add-data "char_by_id.pkl;." \
  --add-data "h2h.pkl;." \
  app.py
```

The output will be in `dist/SpriteClub Predictor/`. Keep all files in the folder together.

---

## How the Model Works

### Features used per match

| Feature | Description |
|---------|-------------|
| Elo | Rating at match time |
| Winstreak | Current consecutive wins/losses |
| Tier | Division ranking (Div1 best → Div6) |
| LifeMax | Character HP |
| Attack / Defense | Base combat stats |
| Crashes | Reliability signal |
| Head-to-head rate | Historical win rate between these two |
| Team aggregates | Avg/min/max across team members |

### Ensemble weights (OVO example)

```
XGBoost       30%
LightGBM      20%
RF per-format 20%
RF global     30%
```

Weights are grid-searched per format on recent held-out data.

### Online learning

Every time you log an outcome, the correction is stored. Every 20 corrections, an XGBoost adapter model is retrained in a background thread and takes over for that format. The more you use it, the more it adapts to the current meta.

---

## Notes

- Character data is from a dataset snapshot. Characters added to SpriteClub after the snapshot date will show as `NOT FOUND`.
- Fuzzy matching is used for near-matches (typos, slight name differences) — these are flagged with `~` in amber.
- The model predicts based on historical patterns. Upsets happen — treat COIN FLIP predictions as a pass.
