name: 백테스트 차트 생성

on:
  workflow_dispatch:

jobs:
  chart:
    runs-on: ubuntu-latest
    timeout-minutes: 60

    steps:
      - uses: actions/checkout@v3

      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: 한글 폰트 설치
        run: |
          sudo apt-get install -y fonts-nanum
          fc-cache -fv

      - name: 패키지 설치
        run: pip install pandas numpy matplotlib yfinance

      - name: matplotlib 캐시 초기화
        run: python3 -c "import matplotlib.font_manager; matplotlib.font_manager._load_fontmanager(try_read_cache=False)"

      - name: 차트 생성
        run: python chart_backtest_kr.py

      - name: PDF 업로드
        uses: actions/upload-artifact@v4
        with:
          name: backtest-charts-kr
          path: backtest_charts_kr.pdf
          retention-days: 30

      - name: 라벨링 CSV 업로드
        uses: actions/upload-artifact@v4
        with:
          name: backtest-label-kr
          path: backtest_label_kr.csv
          retention-days: 30