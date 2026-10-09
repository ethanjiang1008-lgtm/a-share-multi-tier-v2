#!/usr/bin/env python3
"""Standalone V2 daily/intraday scanner.

This helper reuses model feature construction and scoring. During market hours
it overlays today's live quote snapshot on completed daily bars, predicts the
next trading day, and keeps the incomplete bar out of backtest labels. It is
intended for fast manual runs without retraining historical models.
"""
from __future__ import annotations

import datetime
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from zoneinfo import ZoneInfo
from trading_calendar import is_trading_day

from market_data_sina import fetch_all_stocks, fetch_kline, is_main_board
from multi_tier_v2_backtest import (
    KLINE_COUNT,
    MIN_HISTORY,
    WORKERS,
    build_today_prediction,
    market_states,
    choose_live_snapshot_mode,
    build_intraday_snapshot_stock_data,
)

ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/"config"/"models_v2.json"
OUT=ROOT/"data"/"multi_tier_latest.json"

def ensure_intraday_session(now):
    if not is_trading_day(now.date()):
        raise SystemExit(
            f"拒绝执行盘中预测：{now.date()} 不是 A 股交易日。"
            "此次运行不会更改或发布已有预测结果。"
        )
    t=now.time()
    morning=datetime.time(9,30) <= t < datetime.time(11,30)
    afternoon=datetime.time(13,0) <= t < datetime.time(15,0)
    if not (morning or afternoon):
        raise SystemExit(
            f"拒绝执行盘中预测：当前北京时间为 {now.isoformat(timespec='seconds')}。"
            "本任务仅允许在 09:30–11:30 或 13:00–15:00 运行；"
            "盘前、午休、盘后均不会生成或发布交易信号。"
        )


def main():
    run_at=datetime.datetime.now(ZoneInfo("Asia/Shanghai"))
    ensure_intraday_session(run_at)
    quote_rows=fetch_all_stocks()
    universe={
        str(x["code"]):str(x["name"])
        for x in quote_rows
        if is_main_board(str(x.get("code","")),str(x.get("name","")))
        and x.get("code")
    }

    stocks={}
    failed=0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        fs={ex.submit(fetch_kline,c,KLINE_COUNT):c for c in universe}
        for fut in as_completed(fs):
            code=fs[fut]
            try:
                bars=sorted(fut.result(),key=lambda x:str(x.get("date","")))
                if len(bars)>=MIN_HISTORY+22:
                    idx={str(b.get("date")):i for i,b in enumerate(bars)}
                    stocks[code]=(universe[code],idx,bars)
                else:
                    failed+=1
            except Exception:
                failed+=1

    all_dates=sorted({
        datetime.date.fromisoformat(str(b["date"]))
        for _,_,bars in stocks.values()
        for b in bars if b.get("date")
    })
    all_dates=[d for d in all_dates if d<=run_at.date()]
    snapshot_mode=choose_live_snapshot_mode(run_at,all_dates)
    intraday_cutoff=(snapshot_mode=="intraday_snapshot")
    dates=[d for d in all_dates if not (intraday_cutoff and d>=run_at.date())]
    market=market_states(stocks,dates)
    models=json.loads(MODEL_PATH.read_text(encoding="utf-8")) if MODEL_PATH.exists() else {}
    if snapshot_mode:
        prediction_stocks=build_intraday_snapshot_stock_data(
            stocks,quote_rows,run_at.date(),run_at
        )
        prediction_dates=sorted(set(dates)|{run_at.date()})
        prediction_market=market_states(prediction_stocks,prediction_dates)
        payload=build_today_prediction(
            prediction_stocks,models,prediction_dates,prediction_market
        )
    else:
        prediction_stocks=stocks
        payload=build_today_prediction(stocks,models,dates,market)
    if payload:
        payload["analysis_mode"]="盘中实时行情快照（成交量按交易时长估算）"
        payload["run_at"]=run_at.isoformat(timespec="seconds")
        payload["live_quote_count"]=len(prediction_stocks) if snapshot_mode else None
        payload["source_latest_kline_date"]=max(all_dates).isoformat() if all_dates else None

    if payload is None:
        payload={
            "status":"no_data",
            "analysis_date":max(dates).isoformat() if dates else None,
            "prediction_date":None,
            "model_version":"multi-tier-v2",
            "universe":"沪深主板",
            "failed":failed,
            "levels":{str(k):[] for k in range(1,7)}
        }
    else:
        payload["failed"]=failed

    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=="__main__":
    raise SystemExit(main())
