# A股多级连板晋级预测 V2

独立的 V2 结果网页仓库，与 V1 的网页发布相互独立。

## 网页地址

https://ethanjiang1008-lgtm.github.io/a-share-multi-tier-v2/

V1 一进二页面：https://ethanjiang1008-lgtm.github.io/a-share-one-to-two-picker/

## 数据更新机制

此仓库不重新训练模型。V2 模型及历史回测仍由 [原模型仓库](https://github.com/ethanjiang1008-lgtm/a-share-one-to-two-picker) 的工作流计算并提交；本仓库每个工作日北京时间 16:00 从原仓库同步最新的回测报告与候选结果，然后独立部署此站点。也可在 Actions 中手动运行“V2 独立网页发布”。

网页包括各级连板晋级任务的样本外回测、TopK 命中率、AUC、分月表现及最近一次各层级候选。

## 重要说明

- V1 模型与 V1 网站均保留在原仓库；本仓库只负责 V2 页面和页面数据发布。
- 分数用于同一板级内排序。未经独立概率校准，不应把模型分数理解为精确的实际晋级概率。
- 历史回测存在样本量有限及当前股票池回看历史带来的生存者偏差等限制。
