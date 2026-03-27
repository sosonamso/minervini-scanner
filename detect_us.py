"""
미너비니 컵&핸들 공통 모듈 (미국)
scanner_us.py / backtest_us.py 공용
"""
import os, time, requests
import numpy as np
import pandas as pd

# ── 환경변수 ─────────────────────────────────────────────────────
TOK     = os.environ.get("TELEGRAM_TOKEN", "")
CID     = os.environ.get("TELEGRAM_CHAT_ID", "")
MASSIVE = os.environ.get("MASSIVE_TOKEN", "")

# ── 컵&핸들 파라미터 (루즈 - 라벨링 최대 모수 확보) ────────────────
SEARCH_DAYS   = 1250   # 탐색 구간 (~5년)
CUP_MIN_DAYS  = 35     # 컵 최소 너비 (일)
CUP_MAX_DAYS  = 400    # 컵 최대 너비 (일)
CUP_MIN_DEPTH = 0.15   # 컵 최소 깊이 15%
CUP_MAX_DEPTH = 0.60   # 컵 최대 깊이 60%
RH_LH_MAX     = 1.10   # RH <= LH * 이 값
HDL_MIN_DEPTH = 0.05   # 핸들 최소 깊이 5%
HDL_MAX_DEPTH = 0.20   # 핸들 최대 깊이 20%
HDL_MIN_DAYS  = 3      # 핸들 최소 기간
HDL_MAX_DAYS  = 25     # 핸들 최대 기간
HDL_MIN_POS   = 0.50   # 핸들 위치: 컵바닥 50% 이상 회복
VOL_MIN       = 1.20   # 거래량 최소 배율
PIVOT_LOW     = 0.95   # 현재가 >= pivot * 이 값
PIVOT_HIGH    = 1.08   # 현재가 <= pivot * 이 값

# ── 노이즈 필터 (루즈 - 극단적 케이스만 제거) ───────────────────────
F1_MID_RATIO = 0.90   # 컵 중간 1/3 최고가 <= LH * 이 값 (W형 제거)
F2_DAYS      = 10     # 컵 시작 직전 확인 기간
F2_RISE      = 1.30   # 직전 N일 내 급등 배율 초과시 제외
F3_ZONE      = 0.10   # 바닥 체류 구간 (바닥+10% 이내)
F3_MIN_RATIO = 0.15   # 바닥 체류 최소 비율 15%


# ── 유틸리티 ─────────────────────────────────────────────────────

def send(text):
    print(text)
    if TOK:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TOK}/sendMessage",
                data={"chat_id": CID, "text": text}, timeout=10
            )
        except:
            pass


def send_file(filepath, caption=""):
    if TOK:
        try:
            with open(filepath, "rb") as f:
                requests.post(
                    f"https://api.telegram.org/bot{TOK}/sendDocument",
                    data={"chat_id": CID, "caption": caption},
                    files={"document": f}, timeout=30
                )
        except Exception as e:
            print(f"파일 전송 실패: {e}")


def get_massive_ohlcv(ticker, start, end):
    url    = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": MASSIVE}
    for _ in range(2):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(12)
                continue
            if resp.status_code != 200:
                return None
            results = resp.json().get("results", [])
            if not results or len(results) < 100:
                return None
            df = pd.DataFrame(results)
            df["date"] = (
                pd.to_datetime(df["t"], unit="ms")
                .dt.tz_localize("UTC")
                .dt.tz_convert("America/New_York")
                .dt.normalize()
                .dt.tz_localize(None)
            )
            df = df.set_index("date").sort_index()
            df = df.rename(columns={"c": "Close", "v": "Volume"})
            df = df[["Close", "Volume"]].astype(float).dropna()
            return df if len(df) >= 100 else None
        except:
            time.sleep(3)
    return None


def load_tickers():
    try:
        tdf = pd.read_csv("tickers_us.csv", encoding="utf-8-sig")
        result = {}
        for _, row in tdf.iterrows():
            t = str(row["ticker"]).strip().replace(".", "-")
            if not t or t == "nan" or len(t) > 6:
                continue
            result[t] = {
                "cap":      str(row.get("cap",      "SmallCap")),
                "sector":   str(row.get("sector",   "기타")),
                "name":     str(row.get("name",     t)),
                "exchange": str(row.get("exchange", "NYSE")),
            }
        print(f"tickers_us.csv 로드: {len(result)}개")
        return result
    except Exception as e:
        print(f"tickers_us.csv 로드 실패: {e}")
        return {}


# ── 트렌드 템플릿 ─────────────────────────────────────────────────

def check_trend(df):
    """미너비니 트렌드 템플릿 7조건"""
    if len(df) < 200:
        return False
    c    = df["Close"]
    m50  = c.rolling(50).mean()
    m150 = c.rolling(150).mean()
    m200 = c.rolling(200).mean()
    cur  = float(c.iloc[-1])
    a    = float(m50.iloc[-1])
    b    = float(m150.iloc[-1])
    mv   = m200.dropna()
    if len(mv) < 21:
        return False
    d  = float(mv.iloc[-1])
    d1 = float(mv.iloc[-21])
    if any(pd.isna([a, b, d])):
        return False
    lk = c.iloc[-252:] if len(c) >= 252 else c
    return all([
        cur > b and cur > d,     # 현재가 > 150MA, 200MA
        b > d,                   # 150MA > 200MA
        d > d1,                  # 200MA 상승 중
        a > b and a > d,         # 50MA > 150MA, 200MA
        cur > a,                 # 현재가 > 50MA
        cur >= lk.min() * 1.25,  # 52주 저점 +25%
        cur >= lk.max() * 0.70,  # 52주 고점 -30% 이내
    ])


# ── 컵&핸들 감지 ─────────────────────────────────────────────────

def detect(df):
    """
    컵&핸들 패턴 감지
    Returns: (bool, dict)
    dict keys: cd, hd, cdays, hdays, pivot, cur, vr, vs, cup_start, cup_end
    """
    cl  = df["Close"].values.astype(float)
    vl  = df["Volume"].values.astype(float)
    idx = df.index
    n   = len(cl)

    if n < 60:
        return False, {}

    search_n = min(SEARCH_DAYS, n)
    c        = cl[-search_n:]
    v        = vl[-search_n:]
    w        = len(c)
    offset   = n - search_n      # 원본 df 내 슬라이스 시작 위치

    # 로컬 최고점 후보 탐색
    scan_end = w - HDL_MIN_DAYS - CUP_MIN_DAYS
    peak_candidates = []
    for i in range(10, scan_end):
        lo = max(0, i - 10)
        hi = min(w, i + 10)
        if (c[i] == np.max(c[lo:hi]) and
                c[i] == np.max(c[max(0, i-5):min(w, i+5)])):
            peak_candidates.append(i)
    if not peak_candidates:
        peak_candidates = [int(np.argmax(c[:w // 2]))]

    best          = None
    best_cup_days = 0

    for li in peak_candidates:
        lh = c[li]

        # F2: 컵 시작 직전 급등 필터 (스파이크 후 LH 오류 제거)
        if li >= F2_DAYS:
            pre_min = np.min(c[li - F2_DAYS: li])
            if pre_min > 0 and lh / pre_min > F2_RISE:
                continue

        seg = c[li:]
        if len(seg) < CUP_MIN_DAYS + HDL_MIN_DAYS:
            continue

        # 바닥 탐색
        bi       = li + int(np.argmin(seg))
        bot      = c[bi]
        cd       = (lh - bot) / lh
        cup_days = bi - li

        if not (CUP_MIN_DEPTH <= cd <= CUP_MAX_DEPTH):
            continue
        if not (CUP_MIN_DAYS <= cup_days <= CUP_MAX_DAYS):
            continue

        # F1: 컵 내부 중간 1/3 W형 필터
        m1 = li + (bi - li) // 3
        m2 = li + 2 * (bi - li) // 3
        if m2 > m1 and np.max(c[m1:m2]) > lh * F1_MID_RATIO:
            continue

        # F3: 바닥 체류 비율 (U자형 확인)
        dwell = int(np.sum(c[li:bi] <= bot * (1 + F3_ZONE)))
        if dwell / max(cup_days, 1) < F3_MIN_RATIO:
            continue

        # 오른쪽 고점 (RH)
        rc = c[bi:]
        if len(rc) < HDL_MIN_DAYS + 1:
            continue
        ri = bi + int(np.argmax(rc))
        rh = c[ri]

        if rh < lh * 0.85 or rh > lh * RH_LH_MAX:
            continue

        # 핸들
        hnd  = c[ri:]
        hl   = len(hnd)
        if not (HDL_MIN_DAYS <= hl <= HDL_MAX_DAYS):
            continue
        hlow = float(np.min(hnd))
        hd   = (rh - hlow) / rh
        if not (HDL_MIN_DEPTH <= hd <= HDL_MAX_DEPTH):
            continue
        if (hlow - bot) / max(lh - bot, 1e-9) < HDL_MIN_POS:
            continue

        # 현재가 피벗 범위
        cur = cl[-1]
        if not (rh * PIVOT_LOW <= cur <= rh * PIVOT_HIGH):
            continue

        # 거래량
        vr = (float(np.mean(v[-5:])) / float(np.mean(v[-40:-5]))
              if len(v) >= 40 else 1.0)

        full_cup_days = ri - li
        if full_cup_days <= best_cup_days:
            continue

        # 날짜
        try:
            cup_start = idx[offset + li].strftime("%y.%m.%d")
            cup_end   = idx[offset + ri].strftime("%y.%m.%d")
        except:
            cup_start = cup_end = ""

        best_cup_days = full_cup_days
        best = {
            "cd":        round(cd * 100, 1),
            "hd":        round(hd * 100, 1),
            "cdays":     full_cup_days,
            "hdays":     hl,
            "pivot":     round(float(rh), 2),
            "cur":       round(float(cur), 2),
            "vr":        round(vr, 2),
            "vs":        vr >= VOL_MIN,
            "cup_start": cup_start,
            "cup_end":   cup_end,
        }

    if best is None:
        return False, {}
    return True, best


# ── RS / 점수 ────────────────────────────────────────────────────

def calc_rs(df, mkt):
    """RS 상대강도 (SPY 대비)"""
    def p(d, n):
        return float(d["Close"].iloc[-1] / d["Close"].iloc[-n] - 1) if len(d) >= n else 0.0
    w       = [0.4, 0.2, 0.2, 0.2]
    periods = [63, 126, 189, 252]
    s = sum(w[i] * p(df,  periods[i]) for i in range(4))
    m = sum(w[i] * p(mkt, periods[i]) for i in range(4))
    return round((s - m) * 100, 1)


def calc_score(rs, vr, cd, hd):
    """시그널 종합 점수"""
    if rs >= 25:   s_rs = 100
    elif rs >= 15: s_rs = 80
    elif rs >= 10: s_rs = 60
    elif rs >= 5:  s_rs = 40
    else:          s_rs = 20

    if vr >= 3.0:   s_vr = 100
    elif vr >= 2.5: s_vr = 85
    elif vr >= 2.0: s_vr = 70
    elif vr >= 1.7: s_vr = 55
    else:           s_vr = 40

    if 20 <= cd <= 35:                    s_cd = 100
    elif 15 <= cd < 20 or 35 < cd <= 40: s_cd = 75
    elif 40 < cd <= 50:                   s_cd = 50
    else:                                 s_cd = 30

    if 5 <= hd <= 10:   s_hd = 100
    elif 10 < hd <= 12: s_hd = 75
    elif hd > 12:       s_hd = 50
    else:               s_hd = 60

    return round(s_rs * 0.40 + s_vr * 0.35 + s_cd * 0.15 + s_hd * 0.10)
