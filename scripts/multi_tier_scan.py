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

def main():
    run_at=datetime.datetime.now(ZoneInfo("Asia/Shanghai"))
    quote_rows=fetch_all_stocks()
    quote_snapshot_at=datetime.datetime.now(ZoneInfo("Asia/Shanghai"))
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
    snapshot_mode=choose_live_snapshot_mode(quote_snapshot_at,all_dates)
    uses_quote_snapshot=snapshot_mode in ("intraday_snapshot","closing_quote_snapshot")
    dates=[d for d in all_dates if not (uses_quote_snapshot and d>=quote_snapshot_at.date())]
    market=market_states(stocks,dates)
    models=json.loads(MODEL_PATH.read_text(encoding="utf-8")) if MODEL_PATH.exists() else {}
    if uses_quote_snapshot:
        prediction_stocks=build_intraday_snapshot_stock_data(
            stocks,quote_rows,quote_snapshot_at.date(),quote_snapshot_at
        )
        prediction_dates=sorted(set(dates)|{quote_snapshot_at.date()})
        prediction_market=market_states(prediction_stocks,prediction_dates)
        payload=build_today_prediction(
            prediction_stocks,models,prediction_dates,prediction_market
        )
    else:
        prediction_stocks=stocks
        payload=build_today_prediction(stocks,models,dates,market)
    if payload:
        if snapshot_mode=="intraday_snapshot":
            payload["analysis_mode"]="盘中最新行情快照（成交量按交易时长估算）"
        elif snapshot_mode=="closing_quote_snapshot":
            payload["analysis_mode"]="盘后最新报价快照（日K尚未更新）"
        elif is_trading_day(quote_snapshot_at.date()) and quote_snapshot_at.time() < datetime.time(9,30):
            payload["analysis_mode"]="盘前：最近已完成交易日日K"
        elif is_trading_day(quote_snapshot_at.date()) and quote_snapshot_at.time() >= datetime.time(15,0):
            payload["analysis_mode"]="盘后：当日已完成日K"
        elif not is_trading_day(quote_snapshot_at.date()):
            payload["analysis_mode"]="非交易日：最近交易日日K"
        else:
            payload["analysis_mode"]="最近可用日K（非实时报价快照）"
        payload["run_at"]=quote_snapshot_at.isoformat(timespec="seconds")
        payload["quote_snapshot_at"]=quote_snapshot_at.isoformat(timespec="seconds")
        payload["live_quote_count"]=len(prediction_stocks) if uses_quote_snapshot else None
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
