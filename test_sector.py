"""
pykrx 업종분류 테스트
GitHub Actions에서 실행해서 결과 확인
"""
import subprocess
subprocess.run(["pip", "install", "pykrx", "-q"])

from pykrx import stock
from datetime import datetime, timedelta
import pandas as pd

def get_trading_day():
    d = datetime.today()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")

if __name__ == "__main__":
    today = get_trading_day()
    print(f"조회 날짜: {today}")

    # KOSPI 업종분류
    print("\n=== KOSPI 업종분류 ===")
    try:
        df_kospi = stock.get_market_sector_classifications(today, "KOSPI")
        print(f"성공! {len(df_kospi)}개 종목")
        print(f"컬럼: {list(df_kospi.columns)}")
        print(df_kospi.head(10).to_string())

        # 업종명 유니크 목록
        for col in df_kospi.columns:
            if "업종" in col or "sector" in col.lower():
                print(f"\n업종 유니크값({col}): {df_kospi[col].unique()[:20]}")
    except Exception as e:
        print(f"KOSPI 실패: {e}")

    # KOSDAQ 업종분류
    print("\n=== KOSDAQ 업종분류 ===")
    try:
        df_kosdaq = stock.get_market_sector_classifications(today, "KOSDAQ")
        print(f"성공! {len(df_kosdaq)}개 종목")
        print(f"컬럼: {list(df_kosdaq.columns)}")
        print(df_kosdaq.head(10).to_string())
    except Exception as e:
        print(f"KOSDAQ 실패: {e}")

    # 특정 종목 업종 테스트
    print("\n=== 특정 종목 테스트 ===")
    test_tickers = ["005930", "000660", "035720", "263750"]  # 삼성전자, SK하이닉스, 카카오, 펄어비스
    for ticker in test_tickers:
        try:
            info = stock.get_market_sector_classifications(today, "KOSPI")
            if ticker in info.index:
                print(f"{ticker}: {info.loc[ticker].to_dict()}")
            else:
                info2 = stock.get_market_sector_classifications(today, "KOSDAQ")
                if ticker in info2.index:
                    print(f"{ticker}(KOSDAQ): {info2.loc[ticker].to_dict()}")
        except Exception as e:
            print(f"{ticker} 실패: {e}")
