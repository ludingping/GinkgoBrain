"""A 股日频 板块趋势轮动（设计：GinkgoRoad docs/GinkgoBrain/A股日频-板块趋势轮动-设计.md v0.1，2026-09-23）。

纯函数层：features（§3.1/§3.2）、rotation（§3.3–§3.5 槽位状态机 + 通用执行器）、metrics（§1.2 验收）、
ic_probe（§4 L1 / H-B3）、panel（§1.1 BM-S、L0 覆盖）。DB 读取在 scripts/ 里。成本复用 cn_a_trend_base.costs。
"""
