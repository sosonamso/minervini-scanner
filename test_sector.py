"""
네이버 금융 업종 크롤링 테스트
시그널 종목에 대해서만 호출 (소수 종목)
"""
import requests
import re

def get_naver_sector(ticker):
    """네이버 금융에서 업종명 가져오기"""
    url = f"https://finance.naver.com/item/main.naver?code={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        # 업종 정보 파싱
        match = re.search(r'업종명[^"]*"([^"]+)"', resp.text)
        if match:
            return match.group(1)
        # 대안 패턴
        match2 = re.search(r'코스피\s*/\s*([^\n<]+)', resp.text)
        if match2:
            return match2.group(1).strip()
        match3 = re.search(r'코스닥\s*/\s*([^\n<]+)', resp.text)
        if match3:
            return match3.group(1).strip()
        # 더 직접적인 패턴
        match4 = re.search(r'<em class="coinfo_item_title">업종</em>.*?<em class="coinfo_item_value">(.*?)</em>', resp.text, re.DOTALL)
        if match4:
            return match4.group(1).strip()
        return None
    except Exception as e:
        return None

def get_sector_from_krx_index(ticker):
    """KRX 섹터 지수 구성종목으로 업종 파악"""
    # KRX 섹터 지수 코드
    sector_indices = {
        "1028": "IT",
        "1033": "금융",
        "1034": "헬스케어",
        "1035": "소비재",
        "1036": "산업재",
        "1037": "에너지화학",
        "1038": "소재",
        "1039": "유틸리티",
        "1040": "커뮤니케이션",
    }
    # 향후 구현 가능

if __name__ == "__main__":
    test_tickers = [
        ("005930", "삼성전자"),
        ("000660", "SK하이닉스"),
        ("035420", "NAVER"),
        ("035720", "카카오"),
        ("263750", "펄어비스"),
        ("282720", "금양그린파워"),
        ("336260", "두산퓨얼셀"),
        ("000440", "중앙에너비스"),
    ]

    print("=== 네이버 금융 업종 크롤링 테스트 ===\n")
    for ticker, name in test_tickers:
        sector = get_naver_sector(ticker)
        print(f"{ticker} {name}: {sector}")
