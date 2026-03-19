"""
네이버 금융 미국 종목 섹터 테스트
"""
import requests
import re

def get_us_sector_naver(ticker):
    """네이버 금융에서 미국 종목 섹터 가져오기"""
    # 방법 1: 해외주식 검색
    url = f"https://finance.naver.com/world/sise/delayQuote.naver?symbol={ticker}&fdtc=0"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = "utf-8"
        # 업종/섹터 패턴 찾기
        idx = resp.text.find("섹터")
        if idx > 0:
            print(f"  '섹터' 발견 위치 주변: {resp.text[idx-50:idx+100]}")
        idx2 = resp.text.find("업종")
        if idx2 > 0:
            print(f"  '업종' 발견 위치 주변: {resp.text[idx2-50:idx2+100]}")
        idx3 = resp.text.find("sector")
        if idx3 > 0:
            print(f"  'sector' 발견: {resp.text[idx3-50:idx3+100]}")
        return resp.status_code, len(resp.text)
    except Exception as e:
        return None, str(e)

def get_us_sector_financialmodelingprep(ticker):
    """무료 API로 섹터 가져오기 (로그인 불필요)"""
    # Yahoo Finance 비공식 API
    url = f"https://query2.finance.yahoo.com/v11/finance/quoteSummary/{ticker}?modules=summaryProfile"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        sector = data.get("quoteSummary",{}).get("result",[{}])[0].get("summaryProfile",{}).get("sector","")
        industry = data.get("quoteSummary",{}).get("result",[{}])[0].get("summaryProfile",{}).get("industry","")
        return sector, industry
    except Exception as e:
        return None, str(e)

if __name__ == "__main__":
    test_tickers = ["AAPL", "NVDA", "TSLA", "MU", "ELAN", "AMAT"]

    print("=== Yahoo Finance API 테스트 ===")
    for ticker in test_tickers:
        sector, industry = get_us_sector_financialmodelingprep(ticker)
        print(f"{ticker}: {sector} / {industry}")

    print("\n=== 네이버 해외주식 테스트 ===")
    for ticker in test_tickers[:2]:
        print(f"\n{ticker}:")
        code, size = get_us_sector_naver(ticker)
        print(f"  상태: {code}, 크기: {size}")
