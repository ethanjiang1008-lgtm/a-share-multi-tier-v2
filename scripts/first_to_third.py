#!/usr/bin/env python3
"""Independent first-board-to-third-board model and entry-aware historical evaluation."""
from __future__ import annotations
import csv, math, statistics
from collections import defaultdict
from pathlib import Path

TRAIN_END = "2026-07-08"
OOS_START = "2026-07-09"
ENTRY_GAP_MIN_PCT = -2.0
ENTRY_GAP_MAX_PCT = 3.0
MIN_TRAIN_SAMPLES = 100
MIN_POSITIVES = 5
TOP_KS = (1, 2, 3)
FEATURE_NAMES = [
 "ret_1d","ret_3d","ret_5d","volume_5_vs_20","volume_vs_20d",
 "close_above_ma20","ma5_gt_ma10","ma10_gt_ma20","amplitude_1d",
 "limit_up_count_5","limit_up_count_10","days_since_prev_limit_up",
 "body_ratio","upper_wick_ratio","lower_wick_ratio","open_position_in_day",
 "close_position_in_day","close_near_high","intraday_pullback_pct",
 "prior_amplitude_pct","prior_volume_vs_20d","prior_close_near_high",
 "market_first_count","market_2plus_count","market_zt_count",
 "market_same_level_count","market_higher_level_count","prev_market_first_count",
 "prev_market_2plus_count","prev_market_same_level_count","prev_market_higher_level_count",
 "prev_market_1to2_rate","prev_market_level_continuation_rate","market_heat_ratio",
 "market_first_ratio","market_2plus_ratio","market_zt_change","market_2plus_change",
 "market_first_change","same_level_ratio","higher_level_ratio"
]
TARGETS = {
 "reach_3plus":{"label":"首板后连续晋级至三板及以上","field":"y_reach_3plus"},
 "tradeable_3plus":{"label":"次日开盘可参与且最终晋级三板及以上","field":"y_tradeable_3plus"}
}

def _num(x, default=0.0):
 try:
  v=float(x)
  return v if math.isfinite(v) else default
 except (TypeError,ValueError):
  return default

def _one_word(bar, previous_close):
 pc=_num(previous_close)
 if pc<=0:return False
 rg=(_num(bar.get("high"))-_num(bar.get("low")))/pc*100
 op=(_num(bar.get("open"))/pc-1)*100
 cl=(_num(bar.get("close"))/pc-1)*100
 return rg<=0.25 and op>=9.0 and cl>=9.5

def _entry_ok(gap, one_word):
 return gap is not None and ENTRY_GAP_MIN_PCT<=gap<=ENTRY_GAP_MAX_PCT and not one_word

def build_samples(stock_data, dates, market, feature_builder, board_level_fn, pct_fn):
 """T is the first-board signal, T+1 open is the entry proxy, and T+2 is target day."""
 dates=sorted(dates); samples=[]
 for code,(name,idx,bars) in stock_data.items():
  for pos in range(len(dates)-2):
   day, entry_day, target_day=dates[pos:pos+3]
   i=idx.get(day.isoformat()); j=idx.get(entry_day.isoformat()); k=idx.get(target_day.isoformat())
   if i is None or i<60 or j!=i+1 or k!=i+2: continue
   if board_level_fn(bars,i)!=1: continue
   prev_day=dates[pos-1] if pos>0 else None
   feats=feature_builder(bars,i,market,day,1,prev_day)
   if feats is None: continue
   y3=int(board_level_fn(bars,k)>=3)
   pc=_num(bars[i].get("close")); entry_open=_num(bars[j].get("open")); target_close=_num(bars[k].get("close"))
   gap=pct_fn(entry_open,pc) if pc>0 else None
   one_word=_one_word(bars[j],pc)
   eligible=_entry_ok(gap,one_word)
   ret=(target_close/entry_open-1)*100 if entry_open>0 and target_close>0 else None
   samples.append({
    "date":day.isoformat(),"prediction_date":entry_day.isoformat(),"target_date":target_day.isoformat(),
    "code":str(code),"name":str(name),"features":feats,
    "entry_gap_pct":round(gap,4) if gap is not None else None,"one_word_entry":bool(one_word),
    "entry_eligible":bool(eligible),"return_pct":round(ret,4) if ret is not None else None,
    "y_reach_3plus":y3,"y_tradeable_3plus":int(y3 and eligible)
   })
 return samples

def _fit_model(samples,target_name,min_samples=MIN_TRAIN_SAMPLES):
 from sklearn.linear_model import LogisticRegression
 field=TARGETS[target_name]["field"]; samples=list(samples)
 positives=sum(int(s[field]) for s in samples)
 if len(samples)<min_samples or positives<MIN_POSITIVES or len(samples)-positives<MIN_POSITIVES:return None
 means=[]; stds=[]
 for feature in FEATURE_NAMES:
  vals=[_num(s["features"].get(feature)) for s in samples]
  means.append(statistics.fmean(vals) if vals else 0.0)
  sd=statistics.pstdev(vals) if len(vals)>1 else 1.0
  stds.append(sd if sd>1e-9 else 1.0)
 X=[ [(_num(s["features"].get(f))-means[j])/stds[j] for j,f in enumerate(FEATURE_NAMES)] for s in samples]
 y=[int(s[field]) for s in samples]
 clf=LogisticRegression(C=0.3,max_iter=1500,solver="lbfgs")
 clf.fit(X,y)
 return {
  "version":"first-board-to-third-v1","target":target_name,"target_label":TARGETS[target_name]["label"],
  "trained_through":max(s["date"] for s in samples),"labels_observed_through":max(s["target_date"] for s in samples),
  "train_n":len(samples),"positive_n":positives,"positive_rate":positives/len(samples),"C":0.3,
  "feature_names":list(FEATURE_NAMES),"intercept":float(clf.intercept_[0]),
  "features":[{"name":f,"coef":float(clf.coef_[0][j]),"mean":means[j],"std":stds[j]} for j,f in enumerate(FEATURE_NAMES)]
 }

def score_model(model, features):
 if not model:return None
 z=_num(model.get("intercept"))
 for item in model.get("features",[]):
  sd=_num(item.get("std"),1.0)
  if abs(sd)<1e-9:sd=1.0
  z+=_num(item.get("coef"))*((_num(features.get(item.get("name")))-_num(item.get("mean")))/sd)
 z=max(-35.0,min(35.0,z))
 return 1.0/(1.0+math.exp(-z))

def _auc(rows,field):
 y=[int(r[field]) for r in rows]
 if not y or len(set(y))<2:return None
 try:
  from sklearn.metrics import roc_auc_score
  return float(roc_auc_score(y,[r["score"] for r in rows]))
 except Exception:return None

def _metrics(rows,field):
 by_day=defaultdict(list)
 for r in rows:by_day[r["date"]].append(r)
 for d in by_day:by_day[d].sort(key=lambda r:(-_num(r.get("score")),r["code"]))
 top={}; baselines={}
 for k in TOP_KS:
  selected=hits=hit_days=0; expected=0.0; eligible=[]
  for day_rows in by_day.values():
   chosen=day_rows[:k]
   if not chosen:continue
   rate=sum(int(r[field]) for r in day_rows)/len(day_rows)
   selected+=len(chosen); hits+=sum(int(r[field]) for r in chosen)
   hit_days+=int(any(int(r[field]) for r in chosen)); expected+=len(chosen)*rate
   eligible.extend(r for r in chosen if r.get("entry_eligible") and r.get("return_pct") is not None)
  returns=[_num(r.get("return_pct")) for r in eligible]
  top[str(k)]={
   "selected":selected,"hits":hits,"precision":hits/selected if selected else None,
   "day_hit_days":hit_days,"day_hit_rate":hit_days/len(by_day) if by_day else None,
   "entry_eligible_n":len(eligible),"entry_eligible_rate":len(eligible)/selected if selected else None,
   "mean_return_pct":statistics.fmean(returns) if returns else None,
   "median_return_pct":statistics.median(returns) if returns else None,
   "positive_return_rate":sum(x>0 for x in returns)/len(returns) if returns else None
  }
  baselines[str(k)]=expected/selected if selected else None
 labels=[int(r[field]) for r in rows]
 return {
  "selected":len(rows),"oos_days":len(by_day),"positive_n":sum(labels),
  "positive_rate":sum(labels)/len(labels) if labels else None,
  "top1":top["1"],"top2":top["2"],"top3":top["3"],"random_topk_baseline":baselines,
  "oos_auc":_auc(rows,field),
  "oos_brier":statistics.fmean((float(r["score"])-int(r[field]))**2 for r in rows) if rows else None
 }

def _score_oos(samples,target_name):
 field=TARGETS[target_name]["field"]
 oos=[s for s in samples if s["date"]>=OOS_START]
 scored=[]
 for month in sorted({s["date"][:7] for s in oos}):
  month_rows=[s for s in oos if s["date"][:7]==month]
  cutoff=OOS_START if month==OOS_START[:7] else month+"-01"
  # Train only on labels whose outcomes had already occurred; no T+2 leakage.
  train=[s for s in samples if s["target_date"]<cutoff]
  model=_fit_model(train,target_name)
  if not model:continue
  for s in month_rows:
   row=dict(s);row["score"]=score_model(model,s["features"]);scored.append(row)
 return scored

def train_and_backtest(samples):
 samples=list(samples)
 payload={
  "status":"ok","model_version":"first-board-to-third-v1","data_source":"Sina daily K-line",
  "universe":"沪深主板；排除 ST、创业板、科创板、北交所","signal_definition":"分析日为首板日 T",
  "entry_definition":{"day":"T+1 开盘价（历史回测代理，不代表一定成交）","min_open_gap_pct":ENTRY_GAP_MIN_PCT,
   "max_open_gap_pct":ENTRY_GAP_MAX_PCT,"exclude_one_word_board":True},
  "target_definition":"观察 T+2 收盘时是否已连续晋级到三板及以上","feature_names":list(FEATURE_NAMES),"models":{}
 }
 report={
  "status":"ok","model_version":"first-board-to-third-v1","data_source":"Sina daily K-line",
  "universe":payload["universe"],"signal_definition":payload["signal_definition"],
  "entry_definition":payload["entry_definition"],"target_definition":payload["target_definition"],
  "train_end":TRAIN_END,"oos_start":OOS_START,"sample_n":len(samples),
  "labels_observed_through":max((s["target_date"] for s in samples),default=None),"targets":{}
 }
 train_n=sum(s["target_date"]<OOS_START for s in samples)
 for target_name,spec in TARGETS.items():
  final_model=_fit_model(samples,target_name)
  scored=_score_oos(samples,target_name);stats=_metrics(scored,spec["field"])
  full_pos=sum(int(s[spec["field"]]) for s in samples)
  item={
   "label":spec["label"],"sample_n":len(samples),"train_n_before_oos":train_n,
   "oos_n":sum(s["date"]>=OOS_START for s in samples),"oos_scored_n":len(scored),
   "oos_days":stats["oos_days"],"overall_positive_n":full_pos,
   "overall_positive_rate":full_pos/len(samples) if samples else None,
   "oos_positive_n":stats["positive_n"],"oos_baseline":stats["positive_rate"],
   "top1":stats["top1"],"top2":stats["top2"],"top3":stats["top3"],
   "random_topk_baseline":stats["random_topk_baseline"],"oos_auc":stats["oos_auc"],
   "oos_brier":stats["oos_brier"],"monthly":{}
  }
  for month in sorted({r["date"][:7] for r in scored}):
   item["monthly"][month]=_metrics([r for r in scored if r["date"][:7]==month],spec["field"])
  if final_model:
   payload["models"][target_name]=final_model
   item["model_status"]="ready";item["final_model_train_n"]=final_model["train_n"]
   item["final_model_positive_n"]=final_model["positive_n"]
  else:
   payload["models"][target_name]=None;payload["status"]="partial";item["model_status"]="insufficient_samples"
  report["targets"][target_name]=item
 report["target_horizon"]={"signal_day":"T：首板日","entry_day":"T+1：次交易日开盘",
  "evaluation_day":"T+2：第三个交易日（相对首板计第三板）",
  "return_measure":"T+1开盘买入代理价至T+2收盘价的毛收益，不含交易成本与滑点"}
 return payload,report

def save_samples(path,samples):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("w",encoding="utf-8-sig",newline="") as fh:
  w=csv.writer(fh);w.writerow(["date","prediction_date","target_date","code","name","entry_gap_pct",
   "one_word_entry","entry_eligible","return_pct","y_reach_3plus","y_tradeable_3plus"])
  for s in samples:
   w.writerow([s["date"],s["prediction_date"],s["target_date"],s["code"],s["name"],s["entry_gap_pct"],
    s["one_word_entry"],s["entry_eligible"],s["return_pct"],s["y_reach_3plus"],s["y_tradeable_3plus"]])

def build_live_candidates(stock_data,dates,market,model_payload,feature_builder,board_level_fn,
                          analysis_date,prediction_date,next_trading_day_fn,top_n=10):
 from datetime import date as date_type
 try:day=date_type.fromisoformat(str(analysis_date))
 except (TypeError,ValueError):day=max(dates) if dates else None
 if day is None:return {"status":"no_data","candidate_n":0,"rank_reach_3plus":[],"rank_tradeable_3plus":[]}
 ds=sorted(dates);p=ds.index(day) if day in ds else -1;prev=ds[p-1] if p>0 else None
 models=(model_payload or {}).get("models",{});mreach=models.get("reach_3plus");mtrade=models.get("tradeable_3plus")
 rows=[]
 for code,(name,idx,bars) in stock_data.items():
  i=idx.get(day.isoformat())
  if i is None or i<60 or board_level_fn(bars,i)!=1:continue
  feats=feature_builder(bars,i,market,day,1,prev)
  if feats is None:continue
  pr=score_model(mreach,feats);pt=score_model(mtrade,feats)
  if pr is None and pt is None:continue
  rows.append({"code":str(code),"name":str(name),"price":_num(bars[i].get("close")),
   "p_reach_3plus":pr,"p_tradeable_3plus":pt,
   "score_note":"逻辑回归估分；未经独立概率校准，不等同真实概率"})
 reach=sorted(rows,key=lambda r:(-_num(r.get("p_reach_3plus")),r["code"]))[:top_n]
 trade=sorted(rows,key=lambda r:(-_num(r.get("p_tradeable_3plus")),r["code"]))[:top_n]
 for n,row in enumerate(reach,1):row["rank"]=n
 for n,row in enumerate(trade,1):row["rank"]=n
 target_day=None
 try:
  if prediction_date:
   from datetime import date
   target_day=next_trading_day_fn(date.fromisoformat(str(prediction_date))).isoformat()
 except (TypeError,ValueError):pass
 ready=bool(mreach or mtrade)
 return {"status":"ok" if ready else "model_unavailable","model_version":(model_payload or {}).get("model_version","first-board-to-third-v1"),
  "analysis_date":day.isoformat(),"entry_date":prediction_date,"target_3plus_date":target_day,"candidate_n":len(rows),
  "entry_gap_rule":{"min_pct":ENTRY_GAP_MIN_PCT,"max_pct":ENTRY_GAP_MAX_PCT,
   "description":"次日开盘相对首板日收盘价涨跌幅在 -2% 至 +3%，且次日不是一字板"},
  "rank_reach_3plus":reach,"rank_tradeable_3plus":trade}
