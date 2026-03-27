"""
국장 백테스트 차트 생성기
- 데이터: yfinance (.KS/.KQ)
- 출력: backtest_charts_kr.pdf (artifact)
"""
import sys, time, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.font_manager as fm

# 한글 폰트 설정
def set_korean_font():
    import subprocess, os
    # NanumGothic 폰트 경로 탐색
    font_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/nanum/NanumGothic.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            font = fm.FontProperties(fname=fp)
            matplotlib.rcParams["font.family"] = font.get_name()
            matplotlib.rcParams["axes.unicode_minus"] = False
            print(f"폰트 로드: {fp}")
            return
    # fallback
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False
    print("한글 폰트 없음 - DejaVu 사용")

set_korean_font()

CSV_FILE   = "backtest_raw.csv"
OUTPUT_PDF = "backtest_charts_kr.pdf"
SAMPLE_N   = None  # 전체

BG   = "#0d1117"
UP   = "#ef5350"
DN   = "#1565c0"
GRID = "#1e2736"
TXT  = "#e0e0e0"
MA_C = {"MA5":"#f9a825","MA10":"#66bb6a","MA20":"#42a5f5","MA200":"#ce93d8"}
CUP_C = "#26a69a"
SIG_C = "#ef5350"
PIV_C = "#ff9800"


def fetch_ohlcv(ticker, market, start, end):
    suffix = ".KS" if market == "KOSPI" else ".KQ"
    sym = str(ticker).zfill(6) + suffix
    for attempt in range(3):
        try:
            df = yf.download(sym, start=start, end=end,
                             progress=False, auto_adjust=True)
            if df.empty: return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open","High","Low","Close","Volume"]].dropna()
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            print(f"  시도{attempt+1} 실패: {e}")
            time.sleep(3*(attempt+1))
    return None


def add_ma(df):
    for p in [5,10,20,200]:
        df[f"MA{p}"] = df["Close"].rolling(p).mean()
    return df


def nearest_x(dates, dt_str, fmt="%y.%m.%d"):
    try:
        dt = pd.Timestamp(datetime.strptime(str(dt_str), fmt))
        return int(abs(dates - dt).argmin())
    except: return None


def nearest_x2(dates, dt_str, fmt="%Y-%m-%d"):
    try:
        dt = pd.Timestamp(datetime.strptime(str(dt_str), fmt))
        return int(abs(dates - dt).argmin())
    except: return None


def draw(ax_c, ax_v, df, row):
    dates = df.index
    n     = len(dates)
    W     = 0.45

    for i,(_, r) in enumerate(df.iterrows()):
        o,h,l,c = r["Open"],r["High"],r["Low"],r["Close"]
        col = UP if c >= o else DN
        ax_c.plot([i,i],[l,h], color=col, lw=0.7, zorder=2)
        body = max(abs(c-o), (h-l)*0.005)
        rect = Rectangle((i-W/2, min(o,c)), W, body, fc=col, ec=col, zorder=3)
        ax_c.add_patch(rect)

    for ma, col in MA_C.items():
        if ma in df.columns:
            v  = df[ma].values
            ok = ~np.isnan(v)
            if ok.sum() > 1:
                ax_c.plot(np.where(ok)[0], v[ok], color=col,
                          lw=0.9, alpha=0.85, label=ma, zorder=4)

    for i,(_, r) in enumerate(df.iterrows()):
        col = UP if r["Close"] >= r["Open"] else DN
        ax_v.bar(i, r["Volume"], width=0.7, color=col+"99", zorder=2)

    pivot = float(row.get("pivot", 0) or 0)
    stop  = pivot * 0.93 if pivot else 0

    for x, col, ls in [
        (nearest_x(dates, row.get("cup_start","")), CUP_C, "--"),
        (nearest_x(dates, row.get("cup_end","")),   CUP_C, "-"),
        (nearest_x2(dates, row.get("date","")),      SIG_C, "-."),
    ]:
        if x is not None:
            ax_c.axvline(x, color=col, lw=1.3, ls=ls, alpha=0.85, zorder=5)
            ax_v.axvline(x, color=col, lw=1.0, ls=ls, alpha=0.6)

    if pivot:
        ax_c.axhline(pivot, color=PIV_C, lw=1.0, ls=":", alpha=0.9)
        ax_c.text(n-1, pivot, f"  ₩{int(pivot):,}",
                  color=PIV_C, fontsize=7, va="bottom", ha="right")
    if stop:
        ax_c.axhline(stop, color="#ef9a9a", lw=0.8, ls=":", alpha=0.7)
        ax_c.text(n-1, stop, f"  손절₩{int(stop):,}",
                  color="#ef9a9a", fontsize=6.5, va="top", ha="right")

    step = max(1, n//8)
    xtix = list(range(0, n, step))
    ax_c.set_xticks(xtix); ax_c.set_xticklabels([])
    ax_v.set_xticks(xtix)
    ax_v.set_xticklabels([dates[i].strftime("%y.%m") for i in xtix],
                          fontsize=7, color=TXT)
    ax_c.set_xlim(-1, n); ax_v.set_xlim(-1, n)

    pmin = df["Low"].min(); pmax = df["High"].max()
    pad  = (pmax-pmin)*0.07
    ax_c.set_ylim(pmin-pad, pmax+pad*2)
    ax_v.set_ylim(0, df["Volume"].max()*3)
    ax_v.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda x,_: f"{x/1e6:.1f}M" if x>=1e6 else f"{int(x/1000)}K"))

    for ax in [ax_c, ax_v]:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TXT, labelsize=7)
        for sp in ax.spines.values(): sp.set_color(GRID)
        ax.grid(color=GRID, lw=0.4, alpha=0.5)
        ax.yaxis.tick_right()

    ax_c.legend(handles=[
        mpatches.Patch(color=UP, label="양봉"),
        mpatches.Patch(color=DN, label="음봉"),
    ] + [Line2D([0],[0], color=c, lw=1.5, label=m) for m,c in MA_C.items()] + [
        Line2D([0],[0], color=CUP_C, ls="--", lw=1.2, label="컵시작"),
        Line2D([0],[0], color=CUP_C, ls="-",  lw=1.2, label="컵끝"),
        Line2D([0],[0], color=SIG_C, ls="-.", lw=1.5, label="시그널"),
        Line2D([0],[0], color=PIV_C, ls=":",  lw=1.0, label="피벗"),
    ], loc="upper left", fontsize=6.5, ncol=4,
       facecolor="#1a2332", edgecolor=GRID, labelcolor=TXT)


def make_title(row):
    r5  = row.get("r5");  r5s  = f"{r5:+.1f}%"  if pd.notna(r5)  else "-"
    r20 = row.get("r20"); r20s = f"{r20:+.1f}%" if pd.notna(r20) else "-"
    return (
        f"{row['date']}  {row.get('name','')}({row['ticker']})  {row.get('market','')}\n"
        f"점수:{row.get('score','-')}  RS:{row.get('rs','-')}%  "
        f"5일:{r5s}  20일:{r20s}\n"
        f"컵:{row.get('cup_depth','-')}%/{row.get('cup_days','-')}일"
        f"({row.get('cup_start','')}~{row.get('cup_end','')})"
        f"  핸들:{row.get('handle_depth','-')}%/{row.get('handle_days','-')}일"
        f"  거래량:{row.get('vol_ratio','-')}x"
    )


df_bt = pd.read_csv(CSV_FILE).dropna(subset=["date"])

print(f"총 {len(df_bt)}건 차트 생성 시작...")

success = 0
with PdfPages(OUTPUT_PDF) as pdf:
    for i,(_, row) in enumerate(df_bt.iterrows()):
        ticker = str(row["ticker"]).zfill(6)
        market = row.get("market","KOSPI")
        sig_dt = pd.Timestamp(row["date"])
        cup_st = str(row.get("cup_start",""))

        try:
            cs_dt = pd.Timestamp(datetime.strptime(cup_st, "%y.%m.%d"))
            s = (cs_dt - timedelta(days=40)).strftime("%Y-%m-%d")
        except:
            s = (sig_dt - timedelta(days=200)).strftime("%Y-%m-%d")
        e = (sig_dt + timedelta(days=35)).strftime("%Y-%m-%d")

        print(f"[{i+1}/{len(df_bt)}] {ticker} {row.get('name','')} {row['date']}")

        ohlcv = fetch_ohlcv(ticker, market, s, e)
        if ohlcv is None or len(ohlcv) < 5:
            print("  → 데이터 없음, 스킵")
            continue

        ohlcv = add_ma(ohlcv)

        fig = plt.figure(figsize=(14, 8), facecolor=BG)
        gs  = fig.add_gridspec(4, 1, hspace=0.04)
        ax_c = fig.add_subplot(gs[:3, 0])
        ax_v = fig.add_subplot(gs[3, 0])

        try:
            draw(ax_c, ax_v, ohlcv, row)
        except Exception as ex:
            print(f"  → 차트 오류: {ex}")
            plt.close(fig); continue

        fig.text(0.01, 0.99, make_title(row),
                 color=TXT, fontsize=8.5, va="top", ha="left", transform=fig.transFigure,
                 linespacing=1.5)

        pdf.savefig(fig, bbox_inches="tight", facecolor=BG, dpi=120)
        plt.close(fig)
        success += 1
        time.sleep(1.5)

print(f"\n완료: {success}건 → {OUTPUT_PDF}")
