"""
미너비니 컵&핸들 스캐너 - KRX Open API 버전
- 날짜별 전종목 일괄 호출 (하루 2번 = KOSPI + KOSDAQ)
- 420일 × 2 = 840번 호출 (하루 10,000건 한도 내)
"""
import os,time,warnings,requests,json,re
import numpy as np,pandas as pd
from datetime import datetime,timedelta
warnings.filterwarnings("ignore")

TOK=os.environ.get("TELEGRAM_TOKEN","")
CID=os.environ.get("TELEGRAM_CHAT_ID","")
KRX=os.environ.get("KRX_TOKEN","")
SCAN_DAYS=7
HISTORY_DAYS=420
_row_meta={}  # ticker -> {name, sector}

def get_naver_sector_map():
    """네이버 업종번호→업종명 매핑 (스캐너 시작 시 1회 호출)"""
    try:
        url="https://finance.naver.com/sise/sise_group.naver?type=upjong"
        headers={"User-Agent":"Mozilla/5.0"}
        resp=requests.get(url,headers=headers,timeout=15)
        resp.encoding="euc-kr"
        matches=re.findall(r'upjong&amp;no=(\d+)">([^<]+)</a>',resp.text)
        if not matches:
            matches=re.findall(r'upjong&no=(\d+)">([^<]+)</a>',resp.text)
        sector_map={no:name.strip() for no,name in matches if name.strip() and not name.strip().startswith("/")}
        print(f"업종 맵 로딩: {len(sector_map)}개")
        return sector_map
    except Exception as e:
        print(f"업종 맵 실패: {e}")
        return {}

def get_ticker_sector(ticker, sector_map):
    """종목코드 → 업종명"""
    if not sector_map:return "기타"
    try:
        url=f"https://finance.naver.com/item/main.naver?code={ticker}"
        headers={"User-Agent":"Mozilla/5.0"}
        resp=requests.get(url,headers=headers,timeout=10)
        match=re.search(r'type=upjong&no=(\d+)',resp.text)
        if match:
            return sector_map.get(match.group(1),"기타")
    except:pass
    return "기타"

def send_file(filepath, caption=""):
    if TOK:
        try:
            with open(filepath,"rb") as f:
                requests.post(
                    f"https://api.telegram.org/bot{TOK}/sendDocument",
                    data={"chat_id":CID,"caption":caption},
                    files={"document":f},
                    timeout=30
                )
        except Exception as e:
            print(f"파일 전송 실패: {e}")

def send(text):
    print(text)
    if TOK:
        try:requests.post(f"https://api.telegram.org/bot{TOK}/sendMessage",
                          data={"chat_id":CID,"text":text},timeout=10)
        except:pass

# ─────────────────────────────────────
# KRX API 호출
# ─────────────────────────────────────
def get_krx_data(date_str, market="KOSPI"):
    """
    date_str: YYYYMMDD
    market: KOSPI or KOSDAQ
    반환: {ticker: {Open,High,Low,Close,Volume}} dict
    """
    if market=="KOSPI":
        url="https://data-dbg.krx.co.kr/svc/apis/sto/stk_bydd_trd"
    else:
        url="https://data-dbg.krx.co.kr/svc/apis/sto/ksq_bydd_trd"

    headers={
        "AUTH_KEY":KRX.strip(),
        "Content-Type":"application/json",
        "Accept":"application/json",
    }
    payload={"basDd":date_str}

    for attempt in range(3):
        try:
            resp=requests.post(url,headers=headers,json=payload,timeout=30)
            if resp.status_code!=200:
                print(f"KRX API 오류: {resp.status_code} ({market} {date_str})")
                return {}
            data=resp.json()
            block=data.get("OutBlock_1",[])
            if not block:
                # 다른 키도 시도
                for k,v in data.items():
                    if isinstance(v,list) and len(v)>0:
                        block=v
                        break
            if not block:return {}
            result={}
            for row in block:
                try:
                    ticker=str(row.get("ISU_CD","")).strip()
                    if not ticker:continue
                    # 섹터/종목명 저장 (build_ohlcv에서 접근 가능하도록)
                    _row_meta[ticker]={
                        "name":str(row.get("ISU_NM","")).strip(),
                        "sector":str(row.get("SECT_TP_NM","기타")).strip() or "기타"
                    }
                    result[ticker]={
                        "Open":float(str(row.get("TDD_OPNPRC","0")).replace(",","")),
                        "High":float(str(row.get("TDD_HGPRC","0")).replace(",","")),
                        "Low":float(str(row.get("TDD_LWPRC","0")).replace(",","")),
                        "Close":float(str(row.get("TDD_CLSPRC","0")).replace(",","")),
                        "Volume":float(str(row.get("ACC_TRDVOL","0")).replace(",","")),
                        "TrdVal":float(str(row.get("ACC_TRDVAL","0")).replace(",","")),
                    }
                except:pass
            return result
        except Exception as e:
            print(f"KRX 호출 실패(시도{attempt+1}): {e}")
            time.sleep(2*(attempt+1))
    return {}

def get_trading_dates(days=HISTORY_DAYS):
    """최근 N일치 영업일 리스트 (YYYYMMDD 형식)"""
    dates=[]
    d=datetime.today()
    while len(dates)<days:
        if d.weekday()<5:
            dates.append(d.strftime("%Y%m%d"))
        d-=timedelta(days=1)
        if (datetime.today()-d).days>days*2:break
    return list(reversed(dates))  # 오래된 날짜부터

def get_recent_scan_dates(n=7):
    """최근 N 거래일 (탐색용)"""
    dates=[]
    d=datetime.today()
    while len(dates)<n:
        if d.weekday()<5:
            dates.append(d.strftime("%Y-%m-%d"))
        d-=timedelta(days=1)
        if len(dates)>=n*3:break
    return dates[:n]

# ─────────────────────────────────────
# 전체 OHLCV 구축
# ─────────────────────────────────────
def build_ohlcv(trading_dates):
    """
    날짜별 KRX 호출 → 종목별 OHLCV DataFrame 구축
    반환: {ticker: {"market":..., "df":DataFrame}}
    """
    ticker_data={}  # ticker -> {market, rows:[{date,O,H,L,C,V}]}
    ticker_sector={}  # ticker -> 섹터명
    ticker_name={}  # ticker -> 종목명

    total=len(trading_dates)
    for i,date_str in enumerate(trading_dates):
        if i%50==0:print(f"데이터 수집 [{i}/{total}] {date_str}")

        for mkt in ["KOSPI","KOSDAQ"]:
            day_data=get_krx_data(date_str,mkt)
            for ticker,ohlcv in day_data.items():
                if ticker not in ticker_data:
                    ticker_data[ticker]={"market":mkt,"rows":[]}
                ticker_data[ticker]["rows"].append({
                    "date":pd.Timestamp(date_str),
                    **ohlcv
                })
            time.sleep(0.3)  # API 호출 간격

    # DataFrame 변환
    result={}
    for ticker,info in ticker_data.items():
        rows=info["rows"]
        if len(rows)<100:continue
        df=pd.DataFrame(rows).set_index("date").sort_index()
        df=df[["Open","High","Low","Close","Volume","TrdVal"]].astype(float)
        df=df[df["Close"]>0].dropna()
        if len(df)>=100:
            meta=_row_meta.get(ticker,{})
            result[ticker]={"market":info["market"],"df":df,
                           "name":meta.get("name",ticker),
                           "sector":meta.get("sector","기타")}

    return result

# ─────────────────────────────────────
# 미너비니 로직
# ─────────────────────────────────────
def check_market(df):
    """코스피 정배열 확인 (미너비니 정석)
    완전 상승장: 현재가>50MA>150MA>200MA AND 200MA 상승
    부분 상승장: 현재가>200MA
    하락장: 현재가<200MA
    """
    if df is None or len(df)<200:return True,"데이터부족"
    c=df["Close"]
    m50=float(c.rolling(50).mean().iloc[-1])
    m150=float(c.rolling(150).mean().iloc[-1])
    m200=float(c.rolling(200).mean().iloc[-1])
    cur=float(c.iloc[-1])
    m200v=c.rolling(200).mean().dropna()
    m200_21=float(m200v.iloc[-21]) if len(m200v)>=21 else m200
    if any(pd.isna([m50,m150,m200])):return True,"데이터부족"
    if all([cur>m50,m50>m150,m150>m200,m200>m200_21]):
        return True,"완전 상승장(정배열)"
    elif cur>m200:
        return True,"부분 상승장(200MA 위)"
    else:
        return False,"하락장(200MA 하방)"


def check_trend(df):
    if len(df)<200:return False
    c=df["Close"]
    m50=c.rolling(50).mean();m150=c.rolling(150).mean();m200=c.rolling(200).mean()
    cur=float(c.iloc[-1]);a=float(m50.iloc[-1]);b=float(m150.iloc[-1])
    m20v=m200.dropna()
    if len(m20v)<1:return False
    d=float(m20v.iloc[-1]);d1=float(m20v.iloc[-21])if len(m20v)>=21 else d
    if any(pd.isna([a,b,d])):return False
    lk=c.iloc[-252:]if len(c)>=252 else c
    return all([cur>b and cur>d,b>d,d>d1,a>b and a>d,cur>a,
                cur>=lk.min()*1.25,cur>=lk.max()*0.70])

def detect(df):
    cl=df["Close"].values.astype(float);vl=df["Volume"].values.astype(float);n=len(cl)
    if n<60:return False,{}
    c=cl[-min(200,n):];v=vl[-min(200,n):];w=len(c)
    li=int(np.argmax(c[:w//2]));lh=c[li]
    cup=c[li:]
    if len(cup)<20:return False,{}
    bi=li+int(np.argmin(cup));bot=c[bi];cd=(lh-bot)/lh
    if not(0.15<=cd<=0.50)or(bi-li)<35:return False,{}
    rc=c[bi:]
    if len(rc)<10:return False,{}
    ri=bi+int(np.argmax(rc));rh=c[ri]
    if rh<lh*0.90:return False,{}
    hnd=c[ri:];hl=len(hnd)
    if not(5<=hl<=20):return False,{}
    hlow=float(np.min(hnd));hd=(rh-hlow)/rh
    if not(0.05<=hd<=0.15):return False,{}
    if(hlow-bot)/(lh-bot)<0.60:return False,{}
    cur=cl[-1]
    if not(rh*0.97<=cur<=rh*1.05):return False,{}
    vr=float(np.mean(v[-5:]))/float(np.mean(v[-40:-5]))if len(v)>=40 else 1.0
    return True,{"cd":round(cd*100,1),"hd":round(hd*100,1),"cdays":bi-li,"hdays":hl,
                 "pivot":round(float(rh),0),"cur":round(float(cur),0),
                 "vr":round(vr,2),"vs":vr>=1.40}

def calc_rs(df,mkt_df):
    def p(d,n):return float(d["Close"].iloc[-1]/d["Close"].iloc[-n]-1)if len(d)>=n else 0.0
    s=sum([0.4,0.2,0.2,0.2][i]*p(df,[63,126,189,252][i])for i in range(4))
    m=sum([0.4,0.2,0.2,0.2][i]*p(mkt_df,[63,126,189,252][i])for i in range(4))
    return round((s-m)*100,1)

def get_past_signals(df,exclude_ts):
    results=[]
    idx=df.index.tolist()
    check_start=max(0,len(idx)-180)
    for i in range(check_start,len(idx)-5):
        ts=idx[i]
        if ts>=exclude_ts:continue
        sl=df.iloc[:i+1]
        if not check_trend(sl):continue
        ok,pat=detect(sl)
        if not ok:continue
        close_on=float(df["Close"].iloc[i])
        r5=None;r20=None
        if i+5<len(idx):r5=round((float(df["Close"].iloc[i+5])/close_on-1)*100,1)
        if i+20<len(idx):r20=round((float(df["Close"].iloc[i+20])/close_on-1)*100,1)
        results.append({"date":ts.strftime("%y-%m-%d"),"r5":r5,"r20":r20,"vs":pat["vs"]})
    return results


def calc_score(rs, vr, cd, hd):
    """시그널 강도 점수 계산 (백테스트 분포 기반, 0~100점)"""
    # RS 점수 (가중치 40%)
    if rs>=25:s_rs=100
    elif rs>=15:s_rs=80
    elif rs>=10:s_rs=60
    elif rs>=5:s_rs=40
    else:s_rs=20
    # 거래량 배율 점수 (가중치 35%)
    if vr>=3.0:s_vr=100
    elif vr>=2.5:s_vr=85
    elif vr>=2.0:s_vr=70
    elif vr>=1.7:s_vr=55
    else:s_vr=40
    # 컵 깊이 점수 (가중치 15%) — 20~35%가 최적
    if 20<=cd<=35:s_cd=100
    elif 15<=cd<20 or 35<cd<=40:s_cd=75
    elif 40<cd<=50:s_cd=50
    else:s_cd=30
    # 핸들 깊이 점수 (가중치 10%) — 5~10%가 최적
    if 5<=hd<=10:s_hd=100
    elif 10<hd<=12:s_hd=75
    elif hd>12:s_hd=50
    else:s_hd=60
    total=s_rs*0.40+s_vr*0.35+s_cd*0.15+s_hd*0.10
    return round(total)

def score_grade(score):
    if score>=90:return "S"
    elif score>=80:return "A"
    elif score>=70:return "B"
    elif score>=60:return "C"
    else:return "D"

def format_past(history):
    if not history:return ""
    lines=[]
    wins=sum(1 for h in history if h["r20"] is not None and h["r20"]>0)
    total=sum(1 for h in history if h["r20"] is not None)
    for h in history[-3:]:
        r5_s=f"{h['r5']:+.1f}%"if h["r5"] is not None else"-"
        r20_s=f"{h['r20']:+.1f}%"if h["r20"] is not None else"미완"
        ok="OK"if(h["r20"] is not None and h["r20"]>0)else("..."if h["r20"] is None else"XX")
        vol="VOL"if h["vs"]else""
        lines.append(f"  {h['date']}: 5일{r5_s}/20일{r20_s} {ok}{vol}")
    result="[과거이력]\n"+"\n".join(lines)
    if total>0:
        wr=round(wins/total*100)
        result+=f"\n  -> 과거{total}회 승률{wr}%"
    return result

# ─────────────────────────────────────
# 메인
# ─────────────────────────────────────
if __name__=="__main__":
    if not KRX:
        send("KRX_TOKEN이 없어요! GitHub Secrets 확인해주세요.")
        exit(1)

    sig_dates=get_recent_scan_dates(SCAN_DAYS)
    print(f"탐색날짜: {sig_dates}")

    send(f"🚀 KRX API 스캐너 시작\n📅 최근 {SCAN_DAYS}거래일\n({sig_dates[-1]}~{sig_dates[0]})\n{HISTORY_DAYS}일치 데이터 수집 중...\n(약 15~20분 소요)")

    # 데이터 수집
    trading_dates=get_trading_dates(HISTORY_DAYS)
    print(f"수집 대상: {len(trading_dates)}거래일 ({trading_dates[0]}~{trading_dates[-1]})")

    all_ohlcv=build_ohlcv(trading_dates)
    print(f"OHLCV 구축 완료: {len(all_ohlcv)}개 종목")

    # KOSPI + KOSDAQ 지수 수집
    kospi_data={};kosdaq_data={}
    # 지수 대신 ETF 사용 (이름으로 검색)
    kospi_df=None;kosdaq_df=None
    for ticker,info in all_ohlcv.items():
        name=info.get("name","")
        if kospi_df is None and ("KODEX 200" in name or "KODEX200" in name or name=="KODEX 200"):
            kospi_df=info["df"][["Close"]].copy()
            print(f"코스피 지수 ETF: {name}({ticker}) {len(kospi_df)}일치")
        if kosdaq_df is None and ("KODEX 코스닥150" in name or "KODEX코스닥150" in name):
            kosdaq_df=info["df"][["Close"]].copy()
            print(f"코스닥 지수 ETF: {name}({ticker}) {len(kosdaq_df)}일치")
        if kospi_df is not None and kosdaq_df is not None:
            break

    # ETF 못찾으면 이름에 포함된 것으로 재검색
    if kospi_df is None:
        for ticker,info in all_ohlcv.items():
            name=info.get("name","")
            if "KODEX" in name and "200" in name and "레버리지" not in name and "인버스" not in name:
                kospi_df=info["df"][["Close"]].copy()
                print(f"코스피 지수 ETF(대체): {name}({ticker}) {len(kospi_df)}일치")
                break
    if kosdaq_df is None:
        for ticker,info in all_ohlcv.items():
            name=info.get("name","")
            if "KODEX" in name and "코스닥" in name and "레버리지" not in name and "인버스" not in name:
                kosdaq_df=info["df"][["Close"]].copy()
                print(f"코스닥 지수 ETF(대체): {name}({ticker}) {len(kosdaq_df)}일치")
                break

    if kospi_df is None:
        print("코스피 ETF 없음 → 코스피 종목 전체 평균으로 대체")
        kospi_tickers=[t for t,v in all_ohlcv.items() if v["market"]=="KOSPI"][:50]
        closes=pd.concat([all_ohlcv[t]["df"]["Close"].rename(t) for t in kospi_tickers],axis=1)
        kospi_df=closes.mean(axis=1).to_frame("Close")
    if kosdaq_df is None:
        print("코스닥 ETF 없음 → 코스닥 종목 전체 평균으로 대체")
        kosdaq_tickers=[t for t,v in all_ohlcv.items() if v["market"]=="KOSDAQ"][:50]
        closes=pd.concat([all_ohlcv[t]["df"]["Close"].rename(t) for t in kosdaq_tickers],axis=1)
        kosdaq_df=closes.mean(axis=1).to_frame("Close")

    print(f"코스피 지수: {len(kospi_df)}일치")
    print(f"코스닥 지수: {len(kosdaq_df)}일치")
    mkt_df=kospi_df
    print(f"코스피 지수: {len(kospi_df) if kospi_df is not None else 0}일치")
    print(f"코스닥 지수: {len(kosdaq_df) if kosdaq_df is not None else 0}일치")

    market_ok,market_str=check_market(mkt_df)

    # 유효 데이터 통계
    kospi_cnt=sum(1 for v in all_ohlcv.values() if v["market"]=="KOSPI")
    kosdaq_cnt=sum(1 for v in all_ohlcv.values() if v["market"]=="KOSDAQ")
    last_date=trading_dates[-1]

    send(f"데이터 수집 완료\n코스피: {kospi_cnt}개 / 코스닥: {kosdaq_cnt}개\n기준일: {last_date}\n시장: {market_str}\n패턴 분석 시작...")

    if not market_ok:
        send("⚠️ KOSPI 200MA 하방 — 시그널 신뢰도 낮음, 주의!")

    # 패턴 분석
    res=[];trend_pass=0
    ticker_list=list(all_ohlcv.keys())

    for i,ticker in enumerate(ticker_list):
        if i%200==0:print(f"[{i}/{len(ticker_list)}] 트렌드:{trend_pass} 발견:{len(res)}")
        info=all_ohlcv[ticker]
        df=info["df"]
        mkt=info["market"]

        for sig_str in sig_dates:
            sig_ts=pd.Timestamp(sig_str)
            if sig_ts not in df.index:continue
            pos=df.index.tolist().index(sig_ts)
            sl=df.iloc[:pos+1]
            if not check_trend(sl):continue
            trend_pass+=1
            ok,pat=detect(sl)
            if not ok:continue
            if not pat["vs"]:continue
            idx_df=kospi_df if mkt=="KOSPI" else kosdaq_df
            if idx_df is not None and len(idx_df)>10:
                try:rs=calc_rs(sl,idx_df.loc[:sig_ts])
                except:rs=0.0
                if rs<=0:continue
            else:rs=0.0
            history=get_past_signals(df,sig_ts)
            score=calc_score(rs,pat["vr"],pat["cd"],pat["hd"])
            grade=score_grade(score)
            res.append({
                "sig_date":sig_str,"ticker":ticker,
                "name":info.get("name",ticker),
                "sector":info.get("sector","기타"),
                "market":mkt,
                "cur":pat["cur"],"pivot":pat["pivot"],
                "cd":pat["cd"],"hd":pat["hd"],
                "cdays":pat["cdays"],"hdays":pat["hdays"],
                "vr":pat["vr"],"vs":pat["vs"],
                "rs":rs,"score":score,"grade":grade,
                "trdval_20":trdval_20,
                "history":history,
            })
            break

    res.sort(key=lambda x:(x["sig_date"],x["score"],x["rs"]),reverse=True)
    seen=set();deduped=[]
    for r in res:
        if r["ticker"] not in seen:
            seen.add(r["ticker"]);deduped.append(r)
    res=deduped
    print(f"완료: {len(res)}개 발견")

    send(f"스캔 완료\n종목 수: {len(ticker_list)}개\n트렌드 통과: {trend_pass}개\n패턴+거래량+RS: {len(res)}개")

    if not res:
        send(f"📊 미너비니 스캐너\n최근 {SCAN_DAYS}거래일 | {market_str}\n⚠️ 조건 충족 종목 없음")
    else:
        hdr=f"📊 미너비니 컵&핸들\n최근 {SCAN_DAYS}거래일 | {market_str}\n{len(res)}개 발견(RS순)\n"+"─"*24+"\n"
        msg=hdr
        for r in res:
            up=round((r["pivot"]/r["cur"]-1)*100,1)
            mkt_lbl="🔵코스피"if r["market"]=="KOSPI"else"🟢코스닥"
            past=format_past(r["history"])
            grade_emoji={"S":"🏆","A":"🥇","B":"🥈","C":"🥉","D":"📊"}.get(r["grade"],"📊")
            ticker_sector=get_ticker_sector(r["ticker"],sector_map) if sector_map else r.get("sector","기타")
            blk=(f"[{r['sig_date']}] {mkt_lbl} {ticker_sector}\n"
                 f"🔹{r['name']}({r['ticker']})\n"
                 f"  AI점수: {grade_emoji}{r['score']}점({r['grade']}등급)\n"
                 f"  현재가:{r['cur']:,.0f}원\n"
                 f"  피벗:{r['pivot']:,.0f}원({up:+.1f}%)\n"
                 f"  컵:{r['cd']}%/{r['cdays']}일 핸들:{r['hd']}%/{r['hdays']}일\n"
                 f"  거래량:{r['vr']}x🔥 RS:{r['rs']:+.1f}% 거래대금:{r.get('trdval_20',0):.0f}억\n"
                 +(past+"\n" if past else "")+"\n")
            if len(msg)+len(blk)>4000:
                send(msg);msg="(이어서)\n\n"+blk
            else:msg+=blk
        send(msg)

    # RAW 데이터 저장
    if res:
        import pandas as pd
        rows=[]
        for r in res:
            rows.append({
                "date":r["sig_date"],
                "ticker":r["ticker"],
                "name":r.get("name",r["ticker"]),
                "market":r["market"],
                "sector":get_ticker_sector(r["ticker"],sector_map) if sector_map else r.get("sector","기타"),
                "cur":r["cur"],"pivot":r["pivot"],
                "cup_depth":r["cd"],"handle_depth":r["hd"],
                "cup_days":r["cdays"],"handle_days":r["hdays"],
                "vol_ratio":r["vr"],"rs":r["rs"],
                "trdval_20":r.get("trdval_20",0),
                "score":r["score"],"grade":r["grade"],
            })
        pd.DataFrame(rows).to_csv("scanner_kr_raw.csv",index=False,encoding="utf-8-sig")
        print(f"RAW 저장 완료: scanner_kr_raw.csv ({len(rows)}건)")
        send_file("scanner_kr_raw.csv", f"📊 국장 스캐너 RAW ({len(rows)}건) {datetime.today().strftime('%Y-%m-%d')}")
