# cn_a_big_money_rotation

A 股日频 大单净流入横截面轮动策略 v0.1 / v0.2 探索。

**当前状态**：PRELIMINARY · 已关闭（2026-05-13），等扩数据源 / 含熊市样本后复跑。

## 设计文档

- 设计：[GinkgoRoad/docs/GinkgoBrain/A股日频-大单净流入横截面轮动-设计.md](../../../GinkgoRoad/docs/GinkgoBrain/A股日频-大单净流入横截面轮动-设计.md)
- 测试用例：[GinkgoRoad/docs/GinkgoBrain/A股日频-大单净流入横截面轮动-测试用例.md](../../../GinkgoRoad/docs/GinkgoBrain/A股日频-大单净流入横截面轮动-测试用例.md)
- **Verdict 归档**：[GinkgoRoad/docs/GinkgoBrain/A股日频-大单净流入横截面轮动-v0.1-verdict.md](../../../GinkgoRoad/docs/GinkgoBrain/A股日频-大单净流入横截面轮动-v0.1-verdict.md)（**含 3 个 bug 修复记录 + 复跑触发条件**）

## verdict 状态

| 版本 | 信号 | 累积 | Sharpe | MaxDD | vs BM2 | Verdict |
|---|---|---|---|---|---|---|
| v0.1 | 单日 big_net_per_mv | +5.12% | 0.31 | -20.42% | -40.58% | NO-ALPHA |
| v0.2 (default) | 10 日持续吸筹 + 价格温和 + ratio 排序 | +38.55% | 2.33 | -5.75% | -16.82% | NO-ALPHA verdict / 低 beta 防守 portfolio |
| v0.2 [-5%, +15%] | 同上 + 放宽 C3 上限 | +45.85% | 2.16 | -9.15% | -12.76% | NO-ALPHA verdict / 仍跑输 BM2 |
| BM2 等权全市场 | 全持有 | **~55%** | — | — | 0 | — |

13 月样本是连续牛市；v0.2 绝对收益不错但跑输等权，**无法判定是否有真实 alpha 防守特征**——需含熊市样本复跑。

## 一句话概述

T 日盘后用大单资金流向 + 价格做横截面选股，T+1 开盘买入、持有期 ≥3 日；持有期间**不做任何 rebalance**（致命漏洞防线）；卖出全清；双基准（HS300 + 等权全市场）评估、扣成本判 verdict。

## 运行

```bash
# v0.1 baseline（单日 big_net_per_mv）
python -m strategies.cn_a_big_money_rotation.backtest \
    --start 2025-04-02 --end 2026-05-12 --signal v0_1

# v0.2 默认（持续吸筹 + 价格 [-5%, +8%]）
python -m strategies.cn_a_big_money_rotation.backtest \
    --start 2025-04-02 --end 2026-05-12 --signal v0_2

# v0.2 放宽 C3 上限
python -m strategies.cn_a_big_money_rotation.backtest \
    --start 2025-04-02 --end 2026-05-12 --signal v0_2 \
    --v02-price-low -0.05 --v02-price-high 0.15
```

## 数据依赖

- `cnstock_daily_money_flow`（GinkgoSpider 21:38 任务；主键 `(trade_date, stock_code)`）
- `cnstock_daily_snapshot`（GinkgoSpider；最新 1 天快照；**注意字段单位 bug**：见 [verdict §3.1](../../../GinkgoRoad/docs/GinkgoBrain/A股日频-大单净流入横截面轮动-v0.1-verdict.md)）
- `cnstock_security_list`（GinkgoSpider 17:18 任务；活跃股清单）
- `cnstock_kline_day`（已前复权日 K；用于 close 价 + 容量约束基准）

DB 连接走 `utils.db.get_contract_engine()`（cnstock 数据落 `ginkgo_bole` 库）。

## 复跑触发条件

详见 [verdict §6](../../../GinkgoRoad/docs/GinkgoBrain/A股日频-大单净流入横截面轮动-v0.1-verdict.md)。任一满足：
1. Spider 新增数据源（龙虎榜 / 北向资金 / 大宗交易 / 5min money_flow）→ v0.3 用新源
2. `cnstock_daily_money_flow` 数据 ≥18 月且含熊市段 → v0.1.5 复跑验证防守特征
3. 可做空机制（股指期货 / 融券 / ETF 做空通道）→ v0.3 long-short 中性化
