"""A 股日频 大单净流入横截面轮动 v0.1.

设计：GinkgoRoad/docs/GinkgoBrain/A股日频-大单净流入横截面轮动-设计.md
测试用例：同目录 -测试用例.md

模块结构（按 §10）：
- data         : PG → DataFrame；snapshot 反推 float_share + money_flow 因子派生
- universe     : daily_universe(date) 过滤
- signal       : 双信号 + 截面排序 + tie-break + Top20
- portfolio    : 持有期约束、Entry-only rebalance、撮合 + 容量截断
- backtest     : CLI 入口 + vectorized backtester
- evaluate     : IC / 双基准 / verdict 分支
"""

__version__ = "0.1.0-dev"
