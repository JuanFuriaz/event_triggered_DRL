# ---------------------------------------------------------------------------
# Dow Jones 30 stock universe — research list (training start: 2009-01-01)
#
# Decision (PAL consensus GPT-5 8.5/10, Gemini 9/10, Grok-4 9/10):
#   Start from 2009, aligned with FinRL/ESWA standard. The research question
#   is about STATE REPRESENTATIONS, not training period. 11-16 years of data
#   is ample; test windows (COVID 2020, bear 2022) already validate crisis
#   robustness. Starting from 2005 adds complexity without adding signal.
#
# DJIA composition changes 2005-2026 (for paper documentation):
#   2008-09-22  KFT  in, AIG  out (crisis)
#   2009-06-08  TRV+CSCO in, GM+C out (bankruptcy)
#   2012-09-24  UNH  in, KFT  out
#   2013-09-23  GS+NKE+V in, AA+BAC+HPQ out   ← V (Visa) joins DJIA
#   2015-03-19  AAPL in, T   out
#   2018-06-26  WBA  in, GE  out               ← GE leaves; WBA joins
#   2019-04-02  DOW  in, DWDP out (spin-off)
#   2020-08-31  AMGN+HON+CRM in, XOM+PFE+RTX out
#   2024-02-26  AMZN in, WBA out               ← AMZN literally replaces WBA
#   2024-11-08  NVDA+SHW in, INTC+DOW out      ← NVDA literally replaces DOW
#
# Two tickers in FinRL's default list have NO yfinance data from 2009-01-01:
#   - WBA (Walgreens Boots Alliance): ticker created Dec 2014 → NaN 2009-2013
#   - DOW (Dow Inc.):                 spin-off April 2019     → NaN 2009-2018
#   V (Visa, IPO March 2008) is fine from 2009 ✓
#
# Replacements — full-period continuous series, no splicing:
#   Position 26:  WBA → AMZN  (Amazon literally replaced WBA in DJIA Feb 2024)
#   Position 29:  DOW → NVDA  (Nvidia literally replaced DOW in DJIA Nov 2024)
#
# Splicing (e.g., WBA 2014+ | AMZN before) was explicitly rejected: technical
# indicators (SMA, MACD, RSI) generate invalid signals at the splice boundary,
# corrupting the state representation under study. Full replacement is cleaner.
#
# INTC (Intel) kept: in DJIA until Nov 2024, full data from 2009.
# ---------------------------------------------------------------------------

# PRIMARY research list — 30 stocks, clean data from 2009-01-01
# Use: df = YahooDownloader(..., ticker_list=DOW_30_TICKER).fetch_data()
DOW_30_TICKER = [
    "AXP",  "AMGN", "AAPL", "BA",   "CAT",  "CSCO", "CVX",  "GS",
    "HD",   "HON",  "IBM",  "INTC", "JNJ",  "KO",   "JPM",  "MCD",
    "MMM",  "MRK",  "MSFT", "NKE",  "PG",   "TRV",  "UNH",  "CRM",
    "VZ",   "V",    "AMZN", "WMT",  "DIS",  "NVDA",  # pos 26=AMZN, 29=NVDA
]

# FinRL default list (reference only — WBA/DOW have no 2009 data)
DOW_30_FINRL = [
    "AXP",  "AMGN", "AAPL", "BA",   "CAT",  "CSCO", "CVX",  "GS",
    "HD",   "HON",  "IBM",  "INTC", "JNJ",  "KO",   "JPM",  "MCD",
    "MMM",  "MRK",  "MSFT", "NKE",  "PG",   "TRV",  "UNH",  "CRM",
    "VZ",   "V",    "WBA",  "WMT",  "DIS",  "DOW",   # pos 26=WBA, 29=DOW
]

# Swap map: (index, DOW_30_TICKER, DOW_30_FINRL) — same positions
DOW_30_SWAP_MAP = [
    (26, "AMZN", "WBA"),
    (29, "NVDA", "DOW"),
]



import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# ── Seeds ─────────────────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1234]    # 5 seeds per run


FULL_DATE = "2009-01-01_to_2026-01-01"

# ---------------------------------------------------------------------------
# Walk-forward windows — train ends 1 month before test (purge embargo)
# ---------------------------------------------------------------------------
WINDOWS: dict[str, dict[str, tuple[str, str]]] = {
    "w1": {
        "train": ("2009-01-01", "2020-02-01"),   # purge: Feb 2020
        "test":  ("2020-03-01", "2021-01-01"),
    },
    "w2": {
        "train": ("2009-01-01", "2020-12-01"),   # purge: Dec 2020
        "test":  ("2021-01-01", "2022-01-01"),
    },
    "w3": {
        "train": ("2009-01-01", "2024-12-01"),   # purge: Dec 2024
        "test":  ("2025-01-01", "2026-01-01"),
    },
    "w4": {
            "train": ("2009-01-01", "2021-12-01"),   # purge: Dec 2020
            "test":  ("2022-01-01", "2023-01-01"),
    } 
}

TEST_WINDOWS: dict[str, dict[str, tuple[str, str]]] = {
    "w1": {
        "train": ("2009-01-01", "2010-02-01"),   # purge: Feb 2020
        "test":  ("2020-03-01", "2021-01-01"),
    }
}

INDICATORS_MVX = [
    "close_20_sma",
    "close_5_ema",
    "close_5_ema_xu_close_20_sma",
    "close_5_ema_xd_close_20_sma",
    
]

INDICATORS_MV = [
    "close_20_sma",
    "close_5_ema",
]

INDICATORS_MV2 = [
    'close_60_sma',
    'close_30_sma',
]

INDICATORS_MVX2 = [
    'close_60_sma',
    'close_30_sma',
    "close_30_sma_xu_close_60_sma",
    "close_30_sma_xd_close_60_sma",
    
]


INDICATORS_MVX3 = [
    'close_10_sma',
    'close_30_sma',
    "close_10_sma_xu_close_30_sma",
    "close_10_sma_xd_close_30_sma",
    
]

INDICATORS_DBMVX = [
    "close_20_sma",
    "close_5_ema",
    "close_5_ema_xu_close_20_sma",
    "close_5_ema_xd_close_20_sma",
    'close_60_sma',
    'close_30_sma',
    "close_30_sma_xu_close_60_sma",
    "close_30_sma_xd_close_60_sma",
    
]

INDICATORS_BOLLX = [
    "boll_ub",
    "boll_lb",
    "close_xu_boll_lb",
    "close_xu_boll_ub",
    
]

INDICATORS_BOLL = [
    "boll_ub",
    "boll_lb",
    
]

INDICATORS_BOLLMVX = [
    "close_20_sma",
    "close_5_ema",
    "close_5_ema_xu_close_20_sma",
    "close_5_ema_xd_close_20_sma",
    "boll_ub",
    "boll_lb",
    "close_xu_boll_lb",
    "close_xu_boll_ub",
]


INDICATORS_BOLLMVX2 = [
    'close_60_sma',
    'close_30_sma',
    "close_30_sma_xu_close_60_sma",
    "close_30_sma_xd_close_60_sma",
    "boll_ub",
    "boll_lb",
    "close_xu_boll_lb",
    "close_xu_boll_ub",
]



# Phase 2 — Advanced Trigger: Triple EMA Cross (3MVX)
# EMA-5 crosses EMA-10 while EMA-10 > EMA-50
# Compound flag (ema_xu_5_10_50 / ema_xd_5_10_50) computed post-FeatureEngineer
INDICATORS_3MVX1 = [
    "close_5_ema",
    "close_10_ema",
    "close_50_ema",
    "close_5_ema_xu_close_10_ema",
    "close_5_ema_xd_close_10_ema",
    "ema_xu_5_10_50",   # compound: EMA-5 x EMA-10 while EMA-10 > EMA-50
    "ema_xd_5_10_50",
]

INDICATORS_3MVX2 = [
    "close_5_ema",
    "close_10_ema",
    "close_50_ema",
    "close_5_ema_xu_close_50_ema",
    "close_5_ema_xd_close_50_ema",
    "ema_xu_5_10_50",   # compound: EMA-5 x EMA-10 while EMA-10 > EMA-50
    "ema_xd_5_10_50",
]



# Phase 2 — Advanced Trigger: Bollinger Bands + RSI gate (BOLLRSIX)
# Lower band touch + RSI < 35 → long signal
# Upper band touch + RSI > 65 → short signal
# Compound flags (bb_xrsi_long / bb_rsi_short) computed post-FeatureEngineer
INDICATORS_BOLLRSIX1 = [
    "boll_ub",
    "boll_lb",
    "close_xu_boll_lb",
    "close_xd_boll_ub",
    "rsi_14",
    "bb_xrsi_long",     # compound: close_xu_boll_lb AND rsi_14 < 35
    "bb_rsi_short",     # compound: close_xd_boll_ub AND rsi_14 > 65
]

INDICATORS_BOLLRSIX2 = [
    "boll_ub",
    "boll_lb",
    "close_xu_boll_ub",
    "close_xd_boll_lb",
    "rsi_14",
    "bb_breakout_long",   # compound: close_xu_boll_ub AND rsi_14 > 65
    "bb_breakout_short",  # compound: close_xd_boll_lb AND rsi_14 < 35
]

INDICATORS_ALL = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

INDICATORS_MVX2ALL = [
    'close_60_sma',
    'close_30_sma',
    "close_30_sma_xu_close_60_sma",
    "close_30_sma_xd_close_60_sma",
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
]

INDICATORS_MVXALL = [
    "close_20_sma",
    "close_5_ema",
    "close_5_ema_xu_close_20_sma",
    "close_5_ema_xd_close_20_sma",
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
]

INDICATORS_BOLLXALL = [
    "boll_ub",
    "boll_lb",
    "close_xu_boll_lb",
    "close_xu_boll_ub",
    'close_60_sma',
    'close_30_sma',
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    
]

INDICATORS_EMPTY = []

INDICATORS_BASE = {
    "all":       INDICATORS_ALL,
    "mv":        INDICATORS_MV,
    "mv2":       INDICATORS_MV2,
    "mvx":       INDICATORS_MVX,
    "mvx2":      INDICATORS_MVX2,
    "mvx3":      INDICATORS_MVX3,
    "dbmvx":     INDICATORS_DBMVX,
    "boll":      INDICATORS_BOLL,
    "bollx":     INDICATORS_BOLLX,
    "bollmvx":   INDICATORS_BOLLMVX,
    "bollmvx2":  INDICATORS_BOLLMVX2,
    "3mvx1":     INDICATORS_3MVX1,
    "3mvx2":     INDICATORS_3MVX2,
    "bollrsix1": INDICATORS_BOLLRSIX1,
    "bollrsix2": INDICATORS_BOLLRSIX2,
    'mvx2all': INDICATORS_MVX2ALL,
    'mvxall': INDICATORS_MVXALL, 
    'bollxall':INDICATORS_BOLLXALL,
    'empty': INDICATORS_EMPTY
}