# A股多级连板晋级预测 V2

本仓库独立维护 V2 的模型训练、历史回测、每日候选预测和网页发布；不依赖 V1 仓库运行。

## 网页

- V2 独立结果页：https://ethanjiang1008-lgtm.github.io/a-share-multi-tier-v2/
- V1 首板一进二（独立仓库）：https://github.com/ethanjiang1008-lgtm/a-share-one-to-two-picker
- V1 网页：https://ethanjiang1008-lgtm.github.io/a-share-one-to-two-picker/

## 模型任务

| 层级 | 预测任务 |
|---|---|
| 1 → 2 | 首板次日晋级二板 |
| 2 → 3 | 二板次日晋级三板 |
| 3 → 4 | 三板次日晋级四板 |
| 4 → 5 | 四板次日晋级五板 |
| 5 → 6 | 五板次日晋级六板 |
| 6 → 7+ | 六板次日继续晋级 |

每级独立拟合 Logistic 回归模型，使用新浪公开日线数据；股票池限沪深主板，排除 ST、创业板、科创板和北交所。

## 独立运行机制

- `.github/workflows/intraday-scan.yml`：工作日北京时间 14:00 自动运行，生成截至本次运行时刻的最新快照并发布下一实际交易日晋级信号；手动运行允许任何时间触发。盘中使用最新报价快照，盘后优先使用当日已完成日K，日K未更新时使用盘后最新报价快照；盘前/非交易日使用最近一个可用交易日日K，并在网页标注运行模式、数据日期和运行时间。每次新运行会替换网页的最新结果。
- `.github/workflows/model.yml`：仅训练模型和更新历史回测，不生成或覆盖当日交易信号；代码变更时可自动验证/重训，也支持手动运行。
- `.github/workflows/deploy.yml`：在网页前端或历史回测展示数据变更时独立部署 V2 页面。
- V2 的代码、配置、模型、样本和回测均保存在本仓库：
  - `scripts/multi_tier_v2_backtest.py`
  - `scripts/multi_tier_scan.py`
  - `scripts/market_data_sina.py`
  - `scripts/trading_calendar.py`
  - `config/models_v2.json`
  - `reports/multi_tier_v2_backtest.json`
  - `reports/multi_tier_v2_samples.csv`
  - `data/multi_tier_latest.json`

## 指标解释与限制

Top1 / Top2 / Top3 表示每个预测日内按模型评分排名后，选出的前 K 只股票合并计算的实际晋级命中率。模型分数用于同一板级内排序，未经独立校准前不代表精确的实际晋级概率。

历史数据使用当前可获取的主板股票池回看历史，可能存在生存者偏差；高连板层级样本量明显更少，应谨慎解读。历史样本外回测从 2026-07-09 开始，训练数据截止 2026-07-08；新运行产生新结果后，报告会自动更新。
