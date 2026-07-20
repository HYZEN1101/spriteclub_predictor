"""
SpriteClub Predictor v4.3
  • Segoe UI font — clean and readable
  • Dedicated character stats panel with colored text tags
  • Fuzzy match warnings shown in amber with ~ prefix
  • Blue = first team, Red = second team
  • 4-model ensemble + session logger + online learning
"""

import os, sys, re, json, pickle, difflib, datetime, threading
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
import numpy as np
import xgboost as xgb
import lightgbm as lgb

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
def P(f): return os.path.join(BASE, f)

SESSION_FILE = P("sessions.json")
ONLINE_FILE  = P("online_data.npz")
ADAPTED_FILE = P("models_adapted.pkl")

# ── Load models ────────────────────────────────────────────────────────────────
print("Loading models...")
def _load(name):
    with open(P(name),"rb") as f: return pickle.load(f)

_v4      = _load("models_v4.pkl")
_rf_per  = _load("models_per_type.pkl")
_rf_glob = _load("model_v3.pkl")
_weights = _load("ensemble_weights.pkl")
char_by_name = _load("char_by_name.pkl")
char_by_id   = _load("char_by_id.pkl")
h2h_data     = _load("h2h.pkl")
_adapted = _load("models_adapted.pkl") if os.path.exists(ADAPTED_FILE) else {}
print("  All models loaded.")

TIER_ORDER = {"Div1":1,"Div2":2,"Div3":3,"Div4":4,"Div5":5,"Div6":6,
              "New":7,"Untiered":8,"Dupe":9,"Removed":10,"Banned":11}

TIER_COLOR = {
    "Div1":"#ffd700","Div2":"#c0c0c0","Div3":"#cd7f32",
    "Div4":"#7ec8e3","Div5":"#a0a8c0","Div6":"#6a7090",
    "New":"#88bb88","Untiered":"#888","Dupe":"#888","Removed":"#555","Banned":"#e85454",
}

# ── Character lookup — returns (char, was_fuzzy, original_query) ───────────────
def find_char(name):
    key = name.lower().strip()
    if key in char_by_name:
        return char_by_name[key], False, name
    m = difflib.get_close_matches(key, char_by_name.keys(), n=1, cutoff=0.55)
    if m:
        return char_by_name[m[0]], True, name   # (char, fuzzy=True, original)
    return None, False, name

# ── Input parsing ──────────────────────────────────────────────────────────────
_BRACKETS = re.compile(r'\[([^\]]+)\]')
_ARROW    = re.compile(r'\s*[=⇒>]+\s*')

def parse_input(text):
    text = text.strip()
    brackets = _BRACKETS.findall(text)
    if len(brackets) >= 2:
        blue = [n.strip() for n in _ARROW.split(brackets[0]) if n.strip()]
        red  = [n.strip() for n in _ARROW.split(brackets[1]) if n.strip()]
        return blue, red
    vs = re.split(r'\s+[Vv][Ss]?\.?\s+', text)
    if len(vs) == 2:
        return [vs[0].strip()], [vs[1].strip()]
    return None, None

def infer_matchup(a, b):
    la,lb = len(a),len(b)
    if la==1 and lb==1: return "OVO"
    if la==2 and lb==2: return "TVT"
    if la!=lb:           return "TVO"
    if la==3:            return "Turns2"
    if la==4:            return "Turns3"
    return "OVO"

# ── Features ───────────────────────────────────────────────────────────────────
def _tf(chars):
    e,w,t,l,a,d,cr=[],[],[],[],[],[],[]
    for c in chars:
        e.append(c.get("Elo",1500)); w.append(max(-30,min(30,c.get("Winstreak",0))))
        t.append(TIER_ORDER.get(c.get("Tier","New"),7))
        l.append(c.get("LifeMax",1000)); a.append(c.get("Attack",100))
        d.append(c.get("Defense",100)); cr.append(c.get("Crashes",0))
    return (np.mean(e),np.min(e),np.max(e),np.mean(w),np.min(w),np.max(w),
            np.mean(t),np.mean(l),np.mean(a),np.mean(d),np.mean(cr),np.max(cr),len(chars))

def build_features(blue_chars, red_chars):
    L=_tf(blue_chars); R=_tf(red_chars)
    h2h_wr,h2h_n=0.5,0
    if len(blue_chars)==1 and len(red_chars)==1:
        ai,bi=blue_chars[0]["Id"],red_chars[0]["Id"]
        pair=(min(ai,bi),max(ai,bi))
        if pair in h2h_data:
            wins,tot=h2h_data[pair]
            if tot>=3:
                h2h_wr=wins/tot if ai==pair[0] else 1-wins/tot
                h2h_n=min(tot,50)
    return np.array([[
        L[0]-R[0],L[1]-R[1],L[2]-R[2],L[0],R[0],
        L[3]-R[3],L[4]-R[4],L[5]-R[5],L[6]-R[6],
        L[7]-R[7],L[8]-R[8],L[9]-R[9],L[3],R[3],
        h2h_wr,h2h_n,L[10]-R[10],L[11]-R[11],L[12],
    ]],dtype=np.float32)

# ── Ensemble prediction ────────────────────────────────────────────────────────
def predict(blue_chars, red_chars):
    matchup = infer_matchup(blue_chars, red_chars)
    feats   = build_features(blue_chars, red_chars)
    if matchup in _adapted:
        return float(_adapted[matchup].predict_proba(feats)[0][1]), matchup, feats
    w = _weights.get(matchup, {"xgb":0.35,"lgb":0.35,"rf":0.2,"glob":0.1})
    p_xgb  = float(_v4[matchup]['xgb'].predict_proba(feats)[0][1])  if matchup in _v4     else 0.5
    p_lgb  = float(_v4[matchup]['lgb'].predict_proba(feats)[0][1])  if matchup in _v4     else 0.5
    p_rf   = float(_rf_per[matchup].predict_proba(feats)[0][1])     if matchup in _rf_per else 0.5
    p_glob = float(_rf_glob.predict_proba(feats)[0][1])
    prob   = w['xgb']*p_xgb + w['lgb']*p_lgb + w['rf']*p_rf + w['glob']*p_glob
    return prob, matchup, feats

# ── Online learning ────────────────────────────────────────────────────────────
_online_X, _online_y, _online_mt = [], [], []
RETRAIN_EVERY = 20

def _load_online():
    global _online_X,_online_y,_online_mt
    if os.path.exists(ONLINE_FILE):
        d=np.load(ONLINE_FILE,allow_pickle=True)
        _online_X=list(d['X']); _online_y=list(d['y']); _online_mt=list(d['mt'])

def _save_online():
    np.savez(ONLINE_FILE,X=np.array(_online_X,dtype=np.float32),
             y=np.array(_online_y,dtype=np.int8),mt=np.array(_online_mt))

def add_correction(feats, blue_won, matchup):
    _online_X.append(feats[0]); _online_y.append(1 if blue_won else 0)
    _online_mt.append(matchup); _save_online()
    if len(_online_X) % RETRAIN_EVERY == 0:
        threading.Thread(target=_retrain,daemon=True).start(); return True
    return False

def _retrain():
    global _adapted
    mt_arr=np.array(_online_mt); X_arr=np.array(_online_X,dtype=np.float32)
    y_arr=np.array(_online_y,dtype=np.int8)
    for mtype in set(_online_mt):
        mask=mt_arr==mtype; Xm,ym=X_arr[mask],y_arr[mask]
        if len(Xm)<5: continue
        m=xgb.XGBClassifier(n_estimators=50,max_depth=4,learning_rate=0.1,
                              eval_metric='logloss',tree_method='hist',random_state=42,n_jobs=-1)
        m.fit(Xm,ym); _adapted[mtype]=m
    with open(ADAPTED_FILE,"wb") as f: pickle.dump(_adapted,f)

# ── Session ────────────────────────────────────────────────────────────────────
class Session:
    def __init__(self):
        self.active=False; self.bets=[]; self.balance_start=0; self.balance_current=0
        self.start_time=None
        self.history=json.load(open(SESSION_FILE)) if os.path.exists(SESSION_FILE) else []

    def start(self,bal):
        self.active=True; self.bets=[]; self.balance_start=bal
        self.balance_current=bal; self.start_time=datetime.datetime.now().isoformat()

    def log_bet(self,pred_side,actual_side,amount,matchup,blue_chars,red_chars):
        correct=pred_side==actual_side
        profit=amount if correct else -amount
        self.balance_current+=profit
        self.bets.append({"pred":pred_side,"actual":actual_side,"amount":amount,
                           "correct":correct,"profit":profit,"matchup":matchup,
                           "blue":[c['Name'] for c in blue_chars],
                           "red":[c['Name'] for c in red_chars],
                           "time":datetime.datetime.now().isoformat()})
        return correct,profit

    def end(self):
        if not self.active or not self.bets: return None
        total=len(self.bets); correct=sum(1 for b in self.bets if b['correct'])
        profit=self.balance_current-self.balance_start
        rec={"start":self.start_time,"end":datetime.datetime.now().isoformat(),
             "bets":total,"correct":correct,
             "accuracy":round(correct/total*100,1) if total else 0,
             "profit":round(profit,2),"balance_start":self.balance_start,
             "balance_end":round(self.balance_current,2),"detail":self.bets}
        self.history.append(rec)
        with open(SESSION_FILE,"w") as f: json.dump(self.history,f,indent=2)
        self.active=False; return rec

    @property
    def stats(self):
        if not self.bets: return 0,0,0.0,0.0
        total=len(self.bets); correct=sum(1 for b in self.bets if b['correct'])
        return total,correct,correct/total*100,self.balance_current-self.balance_start

session = Session()

# ── Theme ──────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

BG      = "#0f1117"
PANEL   = "#161b27"
CARD    = "#1c2333"
BORDER  = "#252d3d"
ACCENT  = "#e8b84b"
BLUE_C  = "#4a9eed"
BLUE_DIM= "#1e3a5f"
RED_C   = "#e85454"
RED_DIM = "#5f1e1e"
GREEN_C = "#4caf7d"
MUTED   = "#3d4560"
TEXT    = "#dde1ed"
TEXT2   = "#7a8299"
TEXT3   = "#4a5270"
AMBER   = "#f0a830"
FUZZY_C = "#f0a830"   # colour for fuzzy-matched names

# Fonts — Segoe UI for readability, monospace only for numbers
UI      = "Segoe UI"
MONO    = "Consolas"
F_TITLE = (UI, 15, "bold")
F_HEAD  = (UI, 12, "bold")
F_BODY  = (UI, 11)
F_SMALL = (UI, 10)
F_TINY  = (UI, 9)
F_NUM   = (MONO, 11)      # numbers / stats
F_NUM_S = (MONO, 10)

# ── Stat bar helper ────────────────────────────────────────────────────────────
def stat_bar(value, max_val, width=10, fill="█", empty="░"):
    filled = round(value / max_val * width)
    filled = max(0, min(width, filled))
    return fill * filled + empty * (width - filled)

def ws_color(ws):
    if ws >= 5:  return "#4cef90"
    if ws >= 1:  return "#88dd99"
    if ws == 0:  return TEXT2
    if ws >= -4: return "#dd9988"
    return "#e85454"

def elo_color(elo):
    if elo >= 1900: return "#ffd700"
    if elo >= 1700: return "#c0c0c0"
    if elo >= 1500: return "#7ec8e3"
    if elo >= 1300: return "#a0c080"
    return TEXT2

# ── App ────────────────────────────────────────────────────────────────────────
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SpriteClub Predictor")
        self.geometry("1100x820")
        self.minsize(900, 680)
        self.configure(fg_color=BG)
        self._last = None
        _load_online()
        self._build_ui()

    # ══════════════════════════════════════════════════════════════════════════
    #  UI BUILD
    # ══════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        # ── Header ────────────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=52)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="◈  SpriteClub Predictor",
                     font=(UI,16,"bold"), text_color=ACCENT).pack(side="left", padx=20, pady=12)
        ctk.CTkLabel(hdr, text="XGB · LGB · RF × 2  |  4-model ensemble  |  online learning",
                     font=F_TINY, text_color=TEXT3).pack(side="left", pady=12)

        # ── Main layout ────────────────────────────────────────────────────────
        main = ctk.CTkFrame(self, fg_color=BG)
        main.pack(fill="both", expand=True, padx=14, pady=10)
        main.columnconfigure(0, weight=5)
        main.columnconfigure(1, weight=3)
        main.rowconfigure(0, weight=1)

        left  = ctk.CTkFrame(main, fg_color=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        left.rowconfigure(1, weight=1)   # stats panel expands
        left.rowconfigure(2, weight=1)   # result panel expands

        right = ctk.CTkFrame(main, fg_color=BG)
        right.grid(row=0, column=1, sticky="nsew")

        self._build_input(left)
        self._build_stats_panel(left)
        self._build_result(left)
        self._build_right(right)

    # ── Input ─────────────────────────────────────────────────────────────────
    def _build_input(self, parent):
        card = self._card(parent)
        card.grid(row=0, column=0, sticky="ew", pady=(0,8))

        ctk.CTkLabel(card, text="MATCH INPUT", font=F_HEAD, text_color=ACCENT).pack(anchor="w", padx=14, pady=(12,2))
        ctk.CTkLabel(card, text="Paste the SpriteClub chat line, or type  CharA vs CharB",
                     font=F_SMALL, text_color=TEXT2).pack(anchor="w", padx=14, pady=(0,6))

        self.input_box = ctk.CTkTextbox(card, height=52, font=F_BODY,
                                         fg_color=CARD, text_color=TEXT,
                                         border_color=BORDER, border_width=1)
        self.input_box.pack(fill="x", padx=14, pady=(0,8))
        self.input_box.bind("<Return>", lambda e: (self._run_predict(), "break"))

        br = ctk.CTkFrame(card, fg_color="transparent")
        br.pack(fill="x", padx=14, pady=(0,12))
        ctk.CTkButton(br, text="  Predict  ↵  ", command=self._run_predict,
                      font=(UI,12,"bold"), fg_color=ACCENT, text_color="#0d0f14",
                      hover_color="#d4a43c", height=36, corner_radius=8).pack(side="left")
        ctk.CTkButton(br, text="Clear", command=self._clear,
                      font=F_BODY, fg_color=CARD, text_color=TEXT2,
                      hover_color=MUTED, height=36, width=72, corner_radius=8).pack(side="left", padx=(8,0))

    # ── Character stats panel ──────────────────────────────────────────────────
    def _build_stats_panel(self, parent):
        outer = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=10)
        outer.grid(row=1, column=0, sticky="nsew", pady=(0,8))
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(1, weight=1)

        ctk.CTkLabel(outer, text="CHARACTER STATS", font=F_HEAD,
                     text_color=ACCENT).grid(row=0, column=0, columnspan=2,
                                              sticky="w", padx=14, pady=(12,6))

        # Blue side
        blue_frame = ctk.CTkFrame(outer, fg_color=CARD, corner_radius=8)
        blue_frame.grid(row=1, column=0, sticky="nsew", padx=(10,4), pady=(0,10))
        blue_frame.columnconfigure(0, weight=1)
        blue_frame.rowconfigure(0, weight=0)
        blue_frame.rowconfigure(1, weight=1)

        ctk.CTkLabel(blue_frame, text="🔵  BLUE  (first team)",
                     font=(UI,11,"bold"), text_color=BLUE_C).grid(row=0, column=0, sticky="w", padx=10, pady=(8,4))

        self.blue_stats = tk.Text(blue_frame, font=(MONO,10), bg="#1c2333",
                                   fg=TEXT, relief="flat", bd=0,
                                   state="disabled", wrap="none", cursor="arrow",
                                   selectbackground=CARD, highlightthickness=0)
        self.blue_stats.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0,8))
        self._setup_tags(self.blue_stats)

        # Red side
        red_frame = ctk.CTkFrame(outer, fg_color=CARD, corner_radius=8)
        red_frame.grid(row=1, column=1, sticky="nsew", padx=(4,10), pady=(0,10))
        red_frame.columnconfigure(0, weight=1)
        red_frame.rowconfigure(0, weight=0)
        red_frame.rowconfigure(1, weight=1)

        ctk.CTkLabel(red_frame, text="🔴  RED  (second team)",
                     font=(UI,11,"bold"), text_color=RED_C).grid(row=0, column=0, sticky="w", padx=10, pady=(8,4))

        self.red_stats = tk.Text(red_frame, font=(MONO,10), bg="#1c2333",
                                  fg=TEXT, relief="flat", bd=0,
                                  state="disabled", wrap="none", cursor="arrow",
                                  selectbackground=CARD, highlightthickness=0)
        self.red_stats.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0,8))
        self._setup_tags(self.red_stats)

        self._clear_stats()

    def _setup_tags(self, widget):
        """Configure all color tags on a Text widget."""
        widget.tag_configure("gold",    foreground="#ffd700")
        widget.tag_configure("silver",  foreground="#c0c0c0")
        widget.tag_configure("bronze",  foreground="#cd7f32")
        widget.tag_configure("blue_t",  foreground="#7ec8e3")
        widget.tag_configure("muted",   foreground=TEXT2)
        widget.tag_configure("dim",     foreground=TEXT3)
        widget.tag_configure("green",   foreground="#4cef90")
        widget.tag_configure("red_t",   foreground="#e85454")
        widget.tag_configure("amber",   foreground=AMBER)
        widget.tag_configure("white",   foreground=TEXT)
        widget.tag_configure("crash",   foreground="#e85454", font=(MONO,9,"bold"))
        widget.tag_configure("fuzzy",   foreground=AMBER,     font=(MONO,10,"bold"))
        widget.tag_configure("label",   foreground=TEXT2,     font=(MONO,9))
        widget.tag_configure("val",     foreground=TEXT,      font=(MONO,10,"bold"))
        widget.tag_configure("h2h",     foreground="#aaa",    font=(MONO,9))
        # Tier tags
        for tier,col in TIER_COLOR.items():
            widget.tag_configure(f"tier_{tier}", foreground=col, font=(MONO,10,"bold"))

    def _clear_stats(self):
        for w in (self.blue_stats, self.red_stats):
            w.configure(state="normal"); w.delete("1.0","end")
            w.insert("end", "\n  No character loaded yet.", "dim")
            w.configure(state="disabled")

    def _render_char_stats(self, widget, char, fuzzy, original_query, side_color):
        """Write a rich stats block for one character into a Text widget."""
        widget.configure(state="normal")
        widget.delete("1.0","end")

        name  = char['Name']
        tier  = char.get('Tier','?')
        elo   = char.get('Elo', 1500)
        ws    = char.get('Winstreak', 0)
        rw    = char.get('RankedWins', 0)
        rl    = char.get('RankedLosses', 0)
        fw    = char.get('FreeWins', 0)
        fl    = char.get('FreeLosses', 0)
        lm    = char.get('LifeMax', 1000)
        pm    = char.get('PowerMax', 3000)
        atk   = char.get('Attack', 100)
        dfn   = char.get('Defense', 100)
        cr    = char.get('Crashes', 0)
        draws = char.get('Draws', 0)
        total = rw + rl
        wr    = f"{100*rw/total:.1f}%" if total else "N/A"
        ws_s  = f"+{ws}" if ws > 0 else str(ws)
        ws_col= ws_color(ws)

        # Fuzzy match warning line
        if fuzzy:
            widget.insert("end", f"  ⚠ Matched from: ", "amber")
            widget.insert("end", f'"{original_query}"\n', "amber")

        # Name line
        widget.insert("end", f"  {name}\n", "val")

        # Tier + Elo
        widget.insert("end", f"\n  Tier  ", "label")
        tier_tag = f"tier_{tier}" if f"tier_{tier}" in [t for t in widget.tag_names()] else "muted"
        # check if tag exists
        existing = widget.tag_names()
        t_tag = f"tier_{tier}" if f"tier_{tier}" in existing else "muted"
        widget.insert("end", f"{tier:<9}", t_tag)
        widget.insert("end", f"Elo  ", "label")
        ec = "gold" if elo>=1900 else "silver" if elo>=1700 else "blue_t" if elo>=1500 else "muted"
        widget.insert("end", f"{elo}\n", ec)

        # Winstreak bar
        widget.insert("end", f"\n  Win Streak  ", "label")
        bar_val = min(abs(ws), 15)
        bar = stat_bar(bar_val, 15, width=12, fill="▪", empty="·")
        ws_tag = "green" if ws > 0 else "red_t" if ws < 0 else "muted"
        widget.insert("end", f"{ws_s:>4}  ", ws_tag)
        widget.insert("end", f"{bar}\n", ws_tag)

        # W/L
        widget.insert("end", f"\n  Ranked      ", "label")
        widget.insert("end", f"{rw}W ", "green")
        widget.insert("end", f"{rl}L ", "red_t")
        widget.insert("end", f"({wr})\n", "muted")

        widget.insert("end", f"  Free        ", "label")
        total_f = fw+fl
        wr_f = f"{100*fw/total_f:.1f}%" if total_f else "N/A"
        widget.insert("end", f"{fw}W ", "green")
        widget.insert("end", f"{fl}L ", "red_t")
        widget.insert("end", f"({wr_f})\n", "muted")

        if draws: widget.insert("end", f"  Draws       {draws}\n", "muted")

        # Combat stats
        widget.insert("end", f"\n  HP     ", "label")
        hp_bar = stat_bar(lm, 4000, width=10)
        widget.insert("end", f"{lm:<6}", "val")
        widget.insert("end", f"{hp_bar}\n", "blue_t")

        widget.insert("end", f"  Power  ", "label")
        pw_bar = stat_bar(pm, 12000, width=10)
        widget.insert("end", f"{pm:<6}", "val")
        widget.insert("end", f"{pw_bar}\n", "blue_t")

        widget.insert("end", f"  Atk    ", "label")
        a_bar = stat_bar(atk, 300, width=10)
        widget.insert("end", f"{atk:<6}", "val")
        widget.insert("end", f"{a_bar}\n", "amber")

        widget.insert("end", f"  Def    ", "label")
        d_bar = stat_bar(dfn, 300, width=10)
        widget.insert("end", f"{dfn:<6}", "val")
        widget.insert("end", f"{d_bar}\n", "blue_t")

        # Crashes
        if cr > 0:
            widget.insert("end", f"\n  ⚠ Crashes: {cr}  ", "crash")
            widget.insert("end", "(unreliable)\n" if cr > 5 else "\n", "crash")

        widget.configure(state="disabled")

    def _render_team_stats(self, widget, char_list, fuzzy_flags):
        """Render multiple characters stacked in the stats widget."""
        widget.configure(state="normal")
        widget.delete("1.0","end")
        for i, (char, fuzzy, orig) in enumerate(zip(char_list, fuzzy_flags, [c['Name'] for c in char_list])):
            if i > 0:
                widget.insert("end", "\n" + "─"*32 + "\n", "dim")
            # fuzzy_flags[i] is (fuzzy_bool, original_query)
        widget.configure(state="disabled")

    # ── Result panel ───────────────────────────────────────────────────────────
    def _build_result(self, parent):
        card = self._card(parent)
        card.grid(row=2, column=0, sticky="nsew")

        ctk.CTkLabel(card, text="PREDICTION", font=F_HEAD,
                     text_color=ACCENT).pack(anchor="w", padx=14, pady=(12,6))

        self.result_text = tk.Text(card, font=F_BODY, bg=CARD, fg=TEXT,
                                    relief="flat", bd=0, state="disabled",
                                    wrap="word", cursor="arrow",
                                    selectbackground=MUTED, highlightthickness=0,
                                    padx=10, pady=8)
        self.result_text.pack(fill="both", expand=True, padx=14, pady=(0,8))
        self._setup_result_tags()

        # Outcome buttons
        self.outcome_frame = ctk.CTkFrame(card, fg_color="transparent")
        self.outcome_frame.pack(fill="x", padx=14, pady=(0,10))
        ctk.CTkLabel(self.outcome_frame, text="Who actually won?",
                     font=F_SMALL, text_color=TEXT2).pack(anchor="w", pady=(0,4))
        obr = ctk.CTkFrame(self.outcome_frame, fg_color="transparent")
        obr.pack(fill="x")
        self.blue_won_btn = ctk.CTkButton(obr, text="🔵 ...",
                                           command=lambda: self._log_outcome("BLUE"),
                                           font=(UI,11,"bold"), fg_color=BLUE_DIM,
                                           border_color=BLUE_C, border_width=2,
                                           text_color=BLUE_C, hover_color="#1e4a7a",
                                           height=38, corner_radius=8)
        self.blue_won_btn.pack(side="left", expand=True, fill="x", padx=(0,6))
        self.red_won_btn = ctk.CTkButton(obr, text="🔴 ...",
                                          command=lambda: self._log_outcome("RED"),
                                          font=(UI,11,"bold"), fg_color=RED_DIM,
                                          border_color=RED_C, border_width=2,
                                          text_color=RED_C, hover_color="#7a1e1e",
                                          height=38, corner_radius=8)
        self.red_won_btn.pack(side="left", expand=True, fill="x")
        self.outcome_frame.pack_forget()

        # Bet row
        self.bet_frame = ctk.CTkFrame(card, fg_color="transparent")
        self.bet_frame.pack(fill="x", padx=14, pady=(0,12))
        ctk.CTkLabel(self.bet_frame, text="Bet amount:", font=F_SMALL,
                     text_color=TEXT2).pack(side="left")
        self.bet_var = tk.StringVar(value="100")
        ctk.CTkEntry(self.bet_frame, textvariable=self.bet_var, width=90,
                     font=F_NUM, fg_color=CARD, text_color=TEXT,
                     border_color=BORDER).pack(side="left", padx=8)
        self.bet_frame.pack_forget()

    def _setup_result_tags(self):
        t = self.result_text
        t.tag_configure("blue",    foreground=BLUE_C, font=(UI,12,"bold"))
        t.tag_configure("red",     foreground=RED_C,  font=(UI,12,"bold"))
        t.tag_configure("gold",    foreground="#ffd700", font=(UI,13,"bold"))
        t.tag_configure("green",   foreground=GREEN_C,  font=(UI,12,"bold"))
        t.tag_configure("amber",   foreground=AMBER)
        t.tag_configure("muted",   foreground=TEXT2)
        t.tag_configure("dim",     foreground=TEXT3)
        t.tag_configure("win",     foreground=GREEN_C, font=(UI,13,"bold"))
        t.tag_configure("label",   foreground=TEXT2,   font=(UI,10))
        t.tag_configure("sep",     foreground=TEXT3)
        t.tag_configure("correct", foreground=GREEN_C, font=(UI,12,"bold"))
        t.tag_configure("wrong",   foreground=RED_C,   font=(UI,12,"bold"))
        t.tag_configure("normal",  foreground=TEXT,    font=F_BODY)
        t.tag_configure("fuzzy",   foreground=AMBER,   font=(UI,10,"bold"))
        t.tag_configure("notfound",foreground=RED_C,   font=(UI,10,"bold"))

    # ── Right panel ────────────────────────────────────────────────────────────
    def _build_right(self, parent):
        parent.rowconfigure(0, weight=0)
        parent.rowconfigure(1, weight=0)
        parent.rowconfigure(2, weight=0)
        parent.rowconfigure(3, weight=1)

        # Session control
        sc = self._card(parent); sc.grid(row=0, column=0, sticky="ew", pady=(0,8))
        ctk.CTkLabel(sc, text="SESSION", font=F_HEAD, text_color=ACCENT).pack(anchor="w", padx=14, pady=(12,4))
        br = ctk.CTkFrame(sc, fg_color="transparent"); br.pack(fill="x", padx=14)
        ctk.CTkLabel(br, text="Starting balance:", font=F_SMALL, text_color=TEXT2).pack(side="left")
        self.bal_var = tk.StringVar(value="1000")
        ctk.CTkEntry(br, textvariable=self.bal_var, width=90, font=F_NUM,
                     fg_color=CARD, text_color=TEXT, border_color=BORDER).pack(side="left", padx=6)
        sb = ctk.CTkFrame(sc, fg_color="transparent"); sb.pack(fill="x", padx=14, pady=8)
        self.start_btn = ctk.CTkButton(sb, text="Start Session", command=self._start_session,
                                        font=F_BODY, fg_color=GREEN_C, hover_color="#3a9060", height=32, corner_radius=8)
        self.start_btn.pack(side="left", expand=True, fill="x", padx=(0,6))
        self.end_btn   = ctk.CTkButton(sb, text="End Session", command=self._end_session,
                                        font=F_BODY, fg_color=MUTED, hover_color="#5a6280", height=32,
                                        corner_radius=8, state="disabled")
        self.end_btn.pack(side="left", expand=True, fill="x")

        # Live stats
        stc = self._card(parent); stc.grid(row=1, column=0, sticky="ew", pady=(0,8))
        ctk.CTkLabel(stc, text="LIVE STATS", font=F_HEAD, text_color=ACCENT).pack(anchor="w", padx=14, pady=(12,4))
        self.stats_text = ctk.CTkTextbox(stc, height=108, font=F_BODY,
                                          fg_color=CARD, text_color=TEXT, state="disabled")
        self.stats_text.pack(fill="x", padx=14, pady=(0,12))
        self._update_stats()

        # Model status
        mc = self._card(parent); mc.grid(row=2, column=0, sticky="ew", pady=(0,8))
        ctk.CTkLabel(mc, text="MODEL STATUS", font=F_HEAD, text_color=ACCENT).pack(anchor="w", padx=14, pady=(12,4))
        self.model_text = ctk.CTkTextbox(mc, height=68, font=F_SMALL,
                                          fg_color=CARD, text_color=TEXT2, state="disabled")
        self.model_text.pack(fill="x", padx=14, pady=(0,12))
        self._update_model_status()

        # Bet history
        hc = self._card(parent); hc.grid(row=3, column=0, sticky="nsew")
        ctk.CTkLabel(hc, text="BET HISTORY", font=F_HEAD, text_color=ACCENT).pack(anchor="w", padx=14, pady=(12,4))
        self.hist_text = ctk.CTkTextbox(hc, font=F_SMALL, fg_color=CARD, text_color=TEXT, state="disabled")
        self.hist_text.pack(fill="both", expand=True, padx=14, pady=(0,12))

    def _card(self, parent):
        return ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=10,
                             border_color=BORDER, border_width=1)

    # ══════════════════════════════════════════════════════════════════════════
    #  PREDICTION LOGIC
    # ══════════════════════════════════════════════════════════════════════════
    def _run_predict(self):
        raw = self.input_box.get("1.0","end").strip()
        if not raw: return

        blue_names, red_names = parse_input(raw)
        if not blue_names:
            self._set_result_simple("⚠  Could not parse input.\n\nTry:\n  Ryu vs Ken\n  [ Char1 ⇒ Char2 ] Vs. [ Char3 ⇒ Char4 ]")
            return

        # Resolve characters, tracking fuzzy matches
        blue_resolved, red_resolved = [], []   # (char, fuzzy, original)
        not_found = []

        for n in blue_names:
            c, fuzzy, orig = find_char(n)
            if c: blue_resolved.append((c, fuzzy, orig))
            else: not_found.append(n)

        for n in red_names:
            c, fuzzy, orig = find_char(n)
            if c: red_resolved.append((c, fuzzy, orig))
            else: not_found.append(n)

        blue_chars = [r[0] for r in blue_resolved]
        red_chars  = [r[0] for r in red_resolved]

        # ── Render character stats panels ──────────────────────────────────────
        self._render_side_stats(self.blue_stats, blue_resolved)
        self._render_side_stats(self.red_stats,  red_resolved)

        # ── Prediction result text ─────────────────────────────────────────────
        t = self.result_text
        t.configure(state="normal"); t.delete("1.0","end")

        # Not found warnings
        for nf in not_found:
            t.insert("end", f"✗  NOT FOUND: ", "notfound")
            t.insert("end", f'"{nf}"\n', "normal")

        # Fuzzy match notices
        all_resolved = blue_resolved + red_resolved
        for char, fuzzy, orig in all_resolved:
            if fuzzy:
                t.insert("end", f"~  ", "fuzzy")
                t.insert("end", f'"{orig}"', "fuzzy")
                t.insert("end", f"  matched as  ", "muted")
                t.insert("end", f'"{char["Name"]}"\n', "fuzzy")

        if not_found or any(r[1] for r in all_resolved):
            t.insert("end", "\n", "normal")

        if not blue_chars or not red_chars:
            t.insert("end", "Cannot predict — resolve character names first.", "muted")
            t.configure(state="disabled"); return

        prob_blue, matchup, feats = predict(blue_chars, red_chars)
        prob_red = 1 - prob_blue
        pred_side = "BLUE" if prob_blue >= prob_red else "RED"
        win_p     = max(prob_blue, prob_red)

        blue_label = blue_chars[0]['Name'] if len(blue_chars)==1 else f"Blue Team ({len(blue_chars)})"
        red_label  = red_chars[0]['Name']  if len(red_chars)==1  else f"Red Team ({len(red_chars)})"
        win_name   = blue_label if pred_side=="BLUE" else red_label

        self._last = {"prob_blue":prob_blue,"matchup":matchup,"feats":feats,
                      "blue_chars":blue_chars,"red_chars":red_chars,
                      "pred_side":pred_side,"blue_label":blue_label,
                      "red_label":red_label,"win_name":win_name}

        # Format line
        t.insert("end", "Format   ", "label")
        t.insert("end", f"{matchup}\n\n", "normal")

        # H2H
        if len(blue_chars)==1 and len(red_chars)==1:
            bi,ri = blue_chars[0]["Id"],red_chars[0]["Id"]
            pair  = (min(bi,ri),max(bi,ri))
            if pair in h2h_data:
                wins,tot = h2h_data[pair]
                if tot >= 3:
                    bw = wins if bi==pair[0] else tot-wins
                    t.insert("end", "H2H      ", "label")
                    t.insert("end", f"{blue_chars[0]['Name']} ", "blue")
                    t.insert("end", f"{bw}–{tot-bw}", "normal")
                    t.insert("end", f"  {red_chars[0]['Name']}\n\n", "red")

        # Separator
        t.insert("end", "─"*38 + "\n", "sep")

        # Bet on
        t.insert("end", "BET ON   ", "label")
        col = "blue" if pred_side=="BLUE" else "red"
        t.insert("end", f"{'🔵' if pred_side=='BLUE' else '🔴'}  {win_name}\n", col)

        # Probability
        t.insert("end", "\nProb     ", "label")
        t.insert("end", f"{win_p*100:.1f}%\n", "gold")

        # Confidence
        conf,conf_col = (
            ("VERY HIGH ██████", "gold")    if win_p>=0.82 else
            ("HIGH      █████░", "green")   if win_p>=0.72 else
            ("MODERATE  ████░░", "amber")   if win_p>=0.62 else
            ("SLIGHT    ███░░░", "amber")   if win_p>=0.54 else
            ("COIN FLIP ██░░░░", "muted")
        )
        t.insert("end", "Conf     ", "label")
        t.insert("end", f"{conf}\n\n", conf_col)

        # Split
        t.insert("end", "🔵 Blue  ", "blue")
        t.insert("end", f"{prob_blue*100:.1f}%", "normal")
        t.insert("end", "   🔴 Red  ", "red")
        t.insert("end", f"{prob_red*100:.1f}%\n", "normal")

        if matchup in _adapted:
            t.insert("end", "\n[adapted model active]\n", "dim")

        t.configure(state="disabled")

        # Update outcome buttons
        self.blue_won_btn.configure(text=f"🔵  {blue_label}")
        self.red_won_btn.configure(text=f"🔴  {red_label}")
        self.outcome_frame.pack(fill="x", padx=14, pady=(0,10))
        if session.active: self.bet_frame.pack(fill="x", padx=14, pady=(0,12))

    def _render_side_stats(self, widget, resolved_list):
        """Render one or more characters into a stats widget."""
        widget.configure(state="normal")
        widget.delete("1.0","end")

        if not resolved_list:
            widget.insert("end", "\n  No characters loaded.", "dim")
            widget.configure(state="disabled"); return

        for i, (char, fuzzy, orig) in enumerate(resolved_list):
            if i > 0:
                widget.insert("end", "\n" + "╌"*30 + "\n", "dim")
            self._render_char_block(widget, char, fuzzy, orig)

        widget.configure(state="disabled")

    def _render_char_block(self, widget, char, fuzzy, orig):
        name  = char['Name']
        tier  = char.get('Tier','?')
        elo   = char.get('Elo',1500)
        ws    = char.get('Winstreak',0)
        rw    = char.get('RankedWins',0);   rl = char.get('RankedLosses',0)
        fw    = char.get('FreeWins',0);     fl = char.get('FreeLosses',0)
        lm    = char.get('LifeMax',1000);   pm = char.get('PowerMax',3000)
        atk   = char.get('Attack',100);     dfn= char.get('Defense',100)
        cr    = char.get('Crashes',0)
        total = rw+rl
        wr    = f"{100*rw/total:.1f}%" if total else "N/A"
        ws_s  = f"+{ws}" if ws>0 else str(ws)

        # Fuzzy warning
        if fuzzy:
            widget.insert("end", f" ~ \"{orig}\"\n", "fuzzy")
            widget.insert("end", f"   matched as:\n", "label")

        # Name
        widget.insert("end", f" {name}\n", "val")

        # Tier | Elo
        widget.insert("end", f" Tier  ", "label")
        tc = f"tier_{tier}"
        if tc in widget.tag_names():
            widget.insert("end", f"{tier:<9}", tc)
        else:
            widget.insert("end", f"{tier:<9}", "muted")
        widget.insert("end", f"Elo ", "label")
        ec = "gold" if elo>=1900 else "silver" if elo>=1700 else "blue_t" if elo>=1500 else "muted"
        widget.insert("end", f"{elo}\n", ec)

        # Winstreak
        ws_tag = "green" if ws>0 else "red_t" if ws<0 else "muted"
        bar = stat_bar(min(abs(ws),15),15,width=10,fill="▪",empty="·")
        widget.insert("end", f" WS    ", "label")
        widget.insert("end", f"{ws_s:>4}  {bar}\n", ws_tag)

        # W/L
        widget.insert("end", f" Ranked ", "label")
        widget.insert("end", f"{rw}W ", "green")
        widget.insert("end", f"{rl}L ", "red_t")
        widget.insert("end", f"({wr})\n", "muted")

        # Combat stats
        widget.insert("end", f" HP    ", "label")
        widget.insert("end", f"{lm:<6}", "val")
        widget.insert("end", f"{stat_bar(lm,4000,8)}\n", "blue_t")

        widget.insert("end", f" Atk   ", "label")
        widget.insert("end", f"{atk:<6}", "val")
        widget.insert("end", f"{stat_bar(atk,300,8)}\n", "amber")

        widget.insert("end", f" Def   ", "label")
        widget.insert("end", f"{dfn:<6}", "val")
        widget.insert("end", f"{stat_bar(dfn,300,8)}\n", "blue_t")

        if cr > 0:
            widget.insert("end", f" ⚠ Crashes: {cr}\n", "crash")

    # ══════════════════════════════════════════════════════════════════════════
    #  OUTCOME / SESSION
    # ══════════════════════════════════════════════════════════════════════════
    def _log_outcome(self, actual_side):
        if not self._last: return
        p = self._last
        correct   = actual_side == p["pred_side"]
        blue_won  = actual_side == "BLUE"
        retrained = add_correction(p["feats"], blue_won, p["matchup"])

        profit = 0.0
        if session.active:
            try: amount = float(self.bet_var.get())
            except: amount = 0.0
            _, profit = session.log_bet(p["pred_side"],actual_side,amount,
                                         p["matchup"],p["blue_chars"],p["red_chars"])
            self._update_stats(); self._refresh_history()

        winner_name = p["blue_label"] if blue_won else p["red_label"]
        t = self.result_text
        t.configure(state="normal")
        t.insert("end", "\n" + "─"*38 + "\n", "sep")
        if correct:
            t.insert("end", f"✅  Correct — {winner_name} won.", "correct")
        else:
            t.insert("end", f"❌  Wrong — {winner_name} won.", "wrong")
        if profit != 0:
            t.insert("end", f"  ({'+' if profit>0 else ''}{profit:.0f})\n", "green" if profit>0 else "red_t")
        else:
            t.insert("end", "\n", "normal")
        if retrained:
            t.insert("end", "🔄 Model adapting in background...\n", "amber")
        t.configure(state="disabled")

        self.outcome_frame.pack_forget()
        self._update_model_status()
        self._last = None

    def _start_session(self):
        try: bal = float(self.bal_var.get())
        except: bal = 1000.0
        session.start(bal)
        self.start_btn.configure(state="disabled")
        self.end_btn.configure(state="normal", fg_color=RED_C, hover_color="#c43a3a")
        self.bet_frame.pack(fill="x", padx=14, pady=(0,12))
        self._update_stats()

    def _end_session(self):
        rec = session.end()
        self.start_btn.configure(state="normal")
        self.end_btn.configure(state="disabled", fg_color=MUTED, hover_color="#5a6280")
        self.bet_frame.pack_forget()
        if rec:
            messagebox.showinfo("Session Complete",
                f"Bets      :  {rec['bets']}\n"
                f"Correct   :  {rec['correct']}/{rec['bets']}  ({rec['accuracy']}%)\n"
                f"Profit    :  {'+' if rec['profit']>=0 else ''}{rec['profit']:.0f}\n"
                f"Balance   :  {rec['balance_start']} → {rec['balance_end']}")
        self._update_stats(); self._refresh_history()

    # ══════════════════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════════════════
    def _set_result_simple(self, text):
        t = self.result_text
        t.configure(state="normal"); t.delete("1.0","end")
        t.insert("end", text); t.configure(state="disabled")

    def _clear(self):
        self.input_box.delete("1.0","end")
        self._set_result_simple("")
        self._clear_stats()
        self.outcome_frame.pack_forget()
        self._last = None

    def _update_stats(self):
        self.stats_text.configure(state="normal"); self.stats_text.delete("1.0","end")
        if session.active:
            total,correct,acc,profit = session.stats; wrong = total-correct
            lines = [f"Status     ACTIVE 🟢",
                     f"Bets       {total}   ✅ {correct}  ❌ {wrong}",
                     f"Accuracy   {acc:.1f}%",
                     f"Profit     {'+' if profit>=0 else ''}{profit:.0f}",
                     f"Balance    {session.balance_current:.0f}"]
        else:
            h = session.history
            if h:
                tb=sum(s['bets'] for s in h); tc=sum(s['correct'] for s in h)
                tp=sum(s['profit'] for s in h)
                lines=[f"Status      idle",f"Sessions    {len(h)}",
                       f"Total bets  {tb}",
                       f"Accuracy    {tc/tb*100:.1f}%" if tb else f"Accuracy    N/A",
                       f"Profit      {'+' if tp>=0 else ''}{tp:.0f}"]
            else:
                lines=["No sessions yet.","","Start a session to track your bets."]
        self.stats_text.insert("end","\n".join(lines))
        self.stats_text.configure(state="disabled")

    def _update_model_status(self):
        self.model_text.configure(state="normal"); self.model_text.delete("1.0","end")
        n=len(_online_X); nxt=RETRAIN_EVERY-(n%RETRAIN_EVERY) if n else RETRAIN_EVERY
        lines=[f"Ensemble   XGB + LGB + RF-type + RF-global",
               f"Corrections  {n}   (retrain in {nxt})",
               f"Adapted    {', '.join(_adapted.keys()) if _adapted else 'none yet'}"]
        self.model_text.insert("end","\n".join(lines))
        self.model_text.configure(state="disabled")

    def _refresh_history(self):
        self.hist_text.configure(state="normal"); self.hist_text.delete("1.0","end")
        if session.bets:
            for b in reversed(session.bets):
                icon="✅" if b['correct'] else "❌"
                name=b['blue'][0] if b['pred']=="BLUE" else b['red'][0]
                self.hist_text.insert("end",f"{icon}  {b['pred']:<5}  {b['profit']:>+.0f}  {name}\n")
        else:
            for rec in reversed(session.history[-7:]):
                d=rec['start'][:10]
                self.hist_text.insert("end",
                    f"{d}  {rec['bets']} bets  {rec['accuracy']}%  "
                    f"{'+' if rec['profit']>=0 else ''}{rec['profit']:.0f}\n")
        self.hist_text.configure(state="disabled")

if __name__=="__main__":
    app = App()
    app.mainloop()
