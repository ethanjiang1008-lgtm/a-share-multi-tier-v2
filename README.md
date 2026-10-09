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

- `.github/workflows/model.yml`：每个交易日北京时间约 15:20 执行历史回测、更新模型和当日预测，并直接部署 V2 网页；也支持手动运行。
- `.github/workflows/deploy.yml`：仅在网页 HTML/CSS/JS 变更时做轻量静态部署，不从 V1 仓库同步模型或数据。
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
