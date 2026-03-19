"""
KRX 업종분류 데이터 테스트
GitHub Actions에서 실행해서 결과 확인
"""
import requests
from datetime import datetime, timedelta

def get_trading_day():
    d = datetime.today()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")

def get_sector_map(market="STK"):
    """
    market: STK=KOSPI, KSQ=KOSDAQ
    반환: {종목코드: 업종명}
    """
    today = get_trading_day()
    print(f"조회 날짜: {today}, 시장: {market}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "http://data.krx.co.kr/contents/MDC/STAT/standard/MDCSTAT03901.cmd",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    # 1단계: OTP 발급
    otp_resp = requests.post(
        "http://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd",
        data={
            "mktId": market,
            "trdDd": today,
            "money": "1",
            "csvxls_isNo": "false",
            "name": "fileDown",
            "url": "dbms/MDC/STAT/standard/MDCSTAT03901"
        },
        headers=headers,
        timeout=30
    )
    print(f"OTP 응답: {otp_resp.status_code} / {otp_resp.text[:50]}")

    if otp_resp.status_code != 200 or not otp_resp.text.strip():
        print("OTP 발급 실패")
        return {}

    # 2단계: CSV 다운로드
    csv_resp = requests.post(
        "http://data.krx.co.kr/comm/fileDn/download_csv/download.cmd",
        data={"code": otp_resp.text.strip()},
        headers=headers,
        timeout=30
    )
    print(f"CSV 응답: {csv_resp.status_code} / 크기: {len(csv_resp.content)}bytes")

    if csv_resp.status_code != 200:
        print("CSV 다운로드 실패")
        return {}

    # CSV 파싱
    import io, pandas as pd
    try:
        df = pd.read_csv(io.BytesIO(csv_resp.content), encoding="euc-kr")
        print(f"컬럼: {list(df.columns)}")
        print(f"샘플:\n{df.head(5).to_string()}")

        # 종목코드 → 업종명 딕셔너리
        # 컬럼명 확인 후 매핑
        code_col = [c for c in df.columns if "코드" in c or "Code" in c.lower()]
        sector_col = [c for c in df.columns if "업종" in c or "sector" in c.lower()]
        print(f"\n코드 컬럼 후보: {code_col}")
        print(f"업종 컬럼 후보: {sector_col}")

        return df
    except Exception as e:
        print(f"파싱 오류: {e}")
        # 원본 텍스트 일부 출력
        try:
            print(csv_resp.content[:500].decode("euc-kr"))
        except:
            print(csv_resp.content[:500])
        return {}

if __name__ == "__main__":
    print("=== KOSPI 업종분류 테스트 ===")
    kospi = get_sector_map("STK")
    print()
    print("=== KOSDAQ 업종분류 테스트 ===")
    kosdaq = get_sector_map("KSQ")
