"""
네이버 금융 업종 크롤링 테스트 v3
"""
import requests
import re
from html.parser import HTMLParser

def get_naver_sector_map():
    """네이버 업종 목록 전체 가져오기"""
    url = "https://finance.naver.com/sise/sise_group.naver?type=upjong"
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Charset": "euc-kr"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.encoding = "euc-kr"
    text = resp.text

    # 패턴: upjong&no=숫자">업종명</a>
    matches = re.findall(r'upjong&amp;no=(\d+)">([^<]+)</a>', text)
    if not matches:
        matches = re.findall(r'upjong&no=(\d+)">([^<]+)</a>', text)

    sector_map = {}
    for no, name in matches:
        name_clean = name.strip()
        if name_clean and not name_clean.startswith('/'):
            sector_map[no] = name_clean

    print(f"업종 수: {len(sector_map)}개")
    if sector_map:
        print(f"샘플: {dict(list(sector_map.items())[:10])}")
    else:
        # HTML 일부 출력해서 구조 확인
        idx = text.find("upjong")
        if idx > 0:
            print(f"HTML 샘플:\n{text[idx-100:idx+300]}")
    return sector_map

def get_naver_sector(ticker, sector_map):
    url = f"https://finance.naver.com/item/main.naver?code={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.encoding = "euc-kr"
    match = re.search(r'type=upjong&no=(\d+)', resp.text)
    if match:
        no = match.group(1)
        return sector_map.get(no, f"업종{no}")
    return "기타"

if __name__ == "__main__":
    print("업종 목록 로딩...")
    sector_map = get_naver_sector_map()

    if sector_map:
        print("\n=== 종목별 업종 ===")
        for ticker, name in [("263750","펄어비스"),("282720","금양그린파워"),("000440","중앙에너비스")]:
            sector = get_naver_sector(ticker, sector_map)
            print(f"{ticker} {name}: {sector}")
