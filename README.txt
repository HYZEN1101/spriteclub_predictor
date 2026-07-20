====================================================
  SpriteClub Predictor — Full App
====================================================

REQUIREMENTS (run once):
  pip install scikit-learn xgboost lightgbm customtkinter numpy

LAUNCH:
  python app.py          ← Full UI with session logging
  python predict.py      ← CLI fallback (no extra deps)

HOW TO USE THE APP:
  1. Paste the SpriteClub chat line into the input box:
       a [ Char1 => Char2 ] Vs. b [ Char3 => Char4 ]
     or type a simple 1v1:
       Ryu vs Ken
  2. Press PREDICT (or hit Enter)
  3. After the match, click "RED WON" or "BLUE WON"
     → This teaches the model from its mistakes
     → Every 20 corrections triggers a background retrain

SESSION TRACKING:
  • Click "START SESSION" and enter your balance
  • Set bet amount in the bottom-left field
  • Log outcomes after each match
  • Click "END SESSION" for a full report
  • All sessions saved to sessions.json

MODEL:
  • XGBoost + LightGBM ensemble per format
  • OVO (1v1) accuracy on recent data: ~82-83%
  • Time-weighted: recent matches count more
  • Per-format models: OVO, TVT, TVO, Turns2, Turns3
  • Online learning: adapts from your corrections

FILE STRUCTURE:
  app.py              ← main application
  predict.py          ← CLI fallback
  models_v4.pkl       ← XGB+LGB per format
  model_v3.pkl        ← RF fallback
  char_by_name.pkl    ← character lookup
  char_by_id.pkl      ← character by ID
  h2h.pkl             ← head-to-head history
  sessions.json       ← your betting history (auto-created)
  online_data.npz     ← correction samples (auto-created)
  models_adapted.pkl  ← adapted models (auto-created)
====================================================
