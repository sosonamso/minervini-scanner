"""
네이버 금융 업종 크롤링 테스트 v2
업종번호 → 업종명 매핑 후 종목 업종 조회
"""
import requests
import re

def get_naver_sector_map():
    """네이버 업종 목록 전체 가져오기"""
    url = "https://finance.naver.com/sise/sise_group.naver?type=upjong"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    # 업종번호 → 업종명 파싱
    matches = re.findall(r'sise_group_detail\.naver\?type=upjong&no=(\d+).*?<td[^>]*>(.*?)</td>', resp.text, re.DOTALL)
    sector_map = {}
    for no, name in matches:
        name_clean = re.sub(r'<[^>]+>', '', name).strip()
        if name_clean:
            sector_map[no] = name_clean
    return sector_map

def get_naver_sector(ticker, sector_map):
    """종목 업종번호 → 업종명"""
    url = f"https://finance.naver.com/item/main.naver?code={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    match = re.search(r'type=upjong&no=(\d+)', resp.text)
    if match:
        no = match.group(1)
        return sector_map.get(no, f"업종{no}")
    return "기타"

if __name__ == "__main__":
    print("업종 목록 로딩...")
    sector_map = get_naver_sector_map()
    print(f"업종 수: {len(sector_map)}개")
    print(f"샘플: {dict(list(sector_map.items())[:5])}")

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

    print("\n=== 종목별 업종 ===")
    for ticker, name in test_tickers:
        sector = get_naver_sector(ticker, sector_map)
        print(f"{ticker} {name}: {sector}")
