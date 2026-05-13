# cn_a_big_money_rotation

A 股日频 大单净流入横截面轮动策略 v0.1（baseline 验证）。

**当前状态**：实施中（v0.1-dev），未跑 verdict。

## 设计文档

- 设计：[GinkgoRoad/docs/GinkgoBrain/A股日频-大单净流入横截面轮动-设计.md](../../../GinkgoRoad/docs/GinkgoBrain/A股日频-大单净流入横截面轮动-设计.md)
- 测试用例：[GinkgoRoad/docs/GinkgoBrain/A股日频-大单净流入横截面轮动-测试用例.md](../../../GinkgoRoad/docs/GinkgoBrain/A股日频-大单净流入横截面轮动-测试用例.md)

## 一句话概述

T 日盘后用 `(主买大单+主买特大单 净额) / 流通市值` 做截面排序、Top20、T+1 开盘买入、持有期 ≥3 日；
持有期间**不做任何 rebalance**（致命漏洞防线）；卖出全清；
双基准（HS300 + 等权全市场）评估、扣成本后判 verdict；PRELIMINARY 级（13 月样本），数据攒到 18 月后 v0.1.5 复跑。

## verdict 状态

- v0.1：⏳ 实施中（未跑 backtest）

## 运行

```bash
# 跑回测（待 M6 实施）
python -m strategies.cn_a_big_money_rotation.backtest \
    --start 2025-04-01 --end 2026-05-11 --initial-capital 10000000
```

## 数据依赖

- `cnstock_daily_money_flow`（GinkgoSpider 21:38 任务；主键 `(trade_date, stock_code)`）
- `cnstock_daily_snapshot`（GinkgoSpider；最新 1 天快照；用于反推 `float_share`）
- `cnstock_security_list`（GinkgoSpider 17:18 任务；活跃股清单）
- `cnstock_kline_day`（已前复权日 K；用于 close 价 + 容量约束基准）

DB 连接走 `utils.db.get_contract_engine()`（cnstock 数据落 `ginkgo_bole` 库）。
