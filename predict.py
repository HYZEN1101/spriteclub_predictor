#!/usr/bin/env python3
"""
SpriteClub Match Predictor v3
  - Time-weighted training (recent matches weighted higher)
  - Per-format models (OVO/TVT/TVO/Turns)
  - Crash rate, h2h, winstreak, tier, elo features

USAGE:
  python predict.py                     # interactive
  python predict.py "CharA" "CharB"     # quick 1v1

  Paste the SpriteClub chat line:
    a [ Char1 => Char2 ] Vs. b [ Char3 => Char4 ]
"""

import sys, re, pickle, difflib, os
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
def _load(name):
    with open(os.path.join(_dir, name), "rb") as f:
        return pickle.load(f)

fallback_model = _load("model_v3.pkl")
_all_models    = _load("models_v4.pkl")
char_by_name    = _load("char_by_name.pkl")
char_by_id      = _load("char_by_id.pkl")
h2h             = _load("h2h.pkl")

TIER_ORDER = {"Div1":1,"Div2":2,"Div3":3,"Div4":4,"Div5":5,"Div6":6,
              "New":7,"Untiered":8,"Dupe":9,"Removed":10,"Banned":11}

# ── Character lookup ───────────────────────────────────────────────────────────
def find_char(name: str):
    key = name.lower().strip()
    if key in char_by_name:
        return char_by_name[key]
    matches = difflib.get_close_matches(key, char_by_name.keys(), n=3, cutoff=0.55)
    if matches:
        return char_by_name[matches[0]]
    return None

# ── Input parsing ──────────────────────────────────────────────────────────────
_BRACKET_ALL = re.compile(r'\[([^\]]+)\]')
_ARROW       = re.compile(r'\s*[=⇒>]+\s*')

def parse_input(text: str):
    """Returns (a_names, b_names). Raises ValueError on bad format."""
    text = text.strip()
    # Find all [...] groups — take first two regardless of what's around them
    brackets = _BRACKET_ALL.findall(text)
    if len(brackets) >= 2:
        a = [n.strip() for n in _ARROW.split(brackets[0]) if n.strip()]
        b = [n.strip() for n in _ARROW.split(brackets[1]) if n.strip()]
        return a, b
    vs = re.split(r'\s+[Vv][Ss]?\.?\s+', text)
    if len(vs) == 2:
        return [vs[0].strip()], [vs[1].strip()]
    raise ValueError(
        "Could not parse. Try:\n"
        "  1v1  : Ryu vs Ken\n"
        "  Teams: [ Char1 => Char2 ] Vs. [ Char3 => Char4 ]\n"
        "  (or paste the SpriteClub chat line directly)"
    )

# ── Matchup type inference ─────────────────────────────────────────────────────
def infer_matchup(a_chars, b_chars):
    la, lb = len(a_chars), len(b_chars)
    if la == 1 and lb == 1: return "OVO"
    if la == 2 and lb == 2: return "TVT"
    if (la == 1 and lb > 1) or (la > 1 and lb == 1): return "TVO"
    if la == 3 and lb == 3: return "Turns2"
    if la == 4 and lb == 4: return "Turns3"
    return "OVO"  # fallback

# ── Feature building ───────────────────────────────────────────────────────────
def _team_feats(char_list):
    e,w,t,l,a,d,cr = [],[],[],[],[],[],[]
    for c in char_list:
        e.append(c.get("Elo",1500))
        w.append(max(-30, min(30, c.get("Winstreak",0))))
        t.append(TIER_ORDER.get(c.get("Tier","New"),7))
        l.append(c.get("LifeMax",1000))
        a.append(c.get("Attack",100))
        d.append(c.get("Defense",100))
        cr.append(c.get("Crashes",0))
    return (np.mean(e),np.min(e),np.max(e),
            np.mean(w),np.min(w),np.max(w),
            np.mean(t),np.mean(l),np.mean(a),np.mean(d),
            np.mean(cr),np.max(cr),len(char_list))

def _h2h(a_chars, b_chars):
    if len(a_chars)==1 and len(b_chars)==1:
        ai, bi = a_chars[0]["Id"], b_chars[0]["Id"]
        pair = (min(ai,bi), max(ai,bi))
        if pair in h2h:
            wins, total = h2h[pair]
            if total >= 3:
                wr = wins/total if ai==pair[0] else 1-wins/total
                return wr, min(total,50), wins if ai==pair[0] else total-wins, total
    return 0.5, 0, None, None

def build_features(a_chars, b_chars):
    L = _team_feats(a_chars)
    R = _team_feats(b_chars)
    h2h_wr, h2h_n, _, _ = _h2h(a_chars, b_chars)
    return [[
        L[0]-R[0], L[1]-R[1], L[2]-R[2],   # elo diffs
        L[0], R[0],                           # raw elos
        L[3]-R[3], L[4]-R[4], L[5]-R[5],   # winstreak diffs
        L[6]-R[6],                            # tier diff
        L[7]-R[7], L[8]-R[8], L[9]-R[9],   # life/atk/def
        L[3], R[3],                           # raw winstreaks
        h2h_wr, h2h_n,
        L[10]-R[10], L[11]-R[11],            # crash diffs
        L[12],                                # team size
    ]]

# ── Confidence label ───────────────────────────────────────────────────────────
def conf_label(p):
    if p >= 0.82: return "VERY HIGH ██████"
    if p >= 0.72: return "HIGH      █████░"
    if p >= 0.62: return "MODERATE  ████░░"
    if p >= 0.54: return "SLIGHT    ███░░░"
    return              "COIN FLIP ██░░░░"

# ── Display ────────────────────────────────────────────────────────────────────
W = 68
DIV = "─" * W

def _char_line(c):
    ws  = c.get("Winstreak", 0)
    rw  = c.get("RankedWins", 0)
    rl  = c.get("RankedLosses", 0)
    tot = rw + rl
    wr  = f"{100*rw/tot:.0f}%" if tot else "N/A"
    ws_s = f"+{ws}" if ws > 0 else str(ws)
    tier = c.get("Tier","?")
    elo  = c.get("Elo","?")
    cr   = c.get("Crashes",0)
    crash_s = f"  ⚠Crashes:{cr}" if cr > 5 else ""
    return (f"    {c['Name']:<26}  Tier:{tier:<8} Elo:{elo:<6} "
            f"WS:{ws_s:<5} W/L:{rw}/{rl}({wr}){crash_s}")

def predict_from_chars(a_chars, b_chars, a_label="RED", b_label="BLUE"):
    matchup = infer_matchup(a_chars, b_chars)
    model   = _all_models.get(matchup, {}).get("xgb", fallback_model) #  fallback_model)

    print(f"\n{DIV}")
    print(f"{'SPRITECLUB PREDICTOR v3':^{W}}")
    print(DIV)
    print(f"  Format: {matchup}")
    print(f"\n  [{a_label}]")
    for c in a_chars: print(_char_line(c))
    print(f"\n  [{b_label}]")
    for c in b_chars: print(_char_line(c))

    # H2H display
    _, _, a_wins, total = _h2h(a_chars, b_chars)
    if total is not None:
        b_wins = total - a_wins
        print(f"\n  Head-to-head: {a_chars[0]['Name']} {a_wins}–{b_wins} "
              f"{b_chars[0]['Name']}  ({total} matches)")

    feats  = build_features(a_chars, b_chars)
    prob_a = model.predict_proba(feats)[0][1]
    prob_b = 1 - prob_a
    win_p  = max(prob_a, prob_b)
    w_lbl  = a_label if prob_a >= prob_b else b_label
    if len(a_chars) == 1 and len(b_chars) == 1:
        w_name = (a_chars[0] if prob_a >= prob_b else b_chars[0])["Name"]
    else:
        w_name = f"Team {w_lbl}"

    print(f"\n{DIV}")
    print(f"  ► BET ON    : {w_lbl} — {w_name}")
    print(f"    Probability : {win_p*100:.1f}%")
    print(f"    Confidence  : {conf_label(win_p)}")
    print(f"\n    {a_label} win : {prob_a*100:.1f}%   "
          f"{b_label} win : {prob_b*100:.1f}%")
    print(DIV + "\n")

# ── Top-level runner ───────────────────────────────────────────────────────────
def run(text: str):
    try:
        a_names, b_names = parse_input(text)
    except ValueError as e:
        print(f"\n  ERROR: {e}\n"); return

    a_chars, b_chars, not_found = [], [], []
    for name in a_names:
        c = find_char(name)
        if c: a_chars.append(c)
        else: not_found.append(name)
    for name in b_names:
        c = find_char(name)
        if c: b_chars.append(c)
        else: not_found.append(name)

    for nf in not_found:
        print(f'\n  NOT FOUND: "{nf}" — not in dataset.')

    if not a_chars or not b_chars:
        print("  Cannot predict — one or both sides unresolvable.\n")
        return

    predict_from_chars(a_chars, b_chars)

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) == 3:
        run(f"{sys.argv[1]} vs {sys.argv[2]}")
    else:
        print("SpriteClub Predictor v3  |  'quit' to exit")
        print("Paste the chat line, or:  CharA vs CharB\n")
        while True:
            try:
                line = input(">> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!"); break
            if line.lower() in ("quit","exit","q"): break
            if line: run(line)
