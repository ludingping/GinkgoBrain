.PHONY: help install test train-stock train-crypto eval-stock eval-crypto tb clean

PYTHON     := python
CONFIG     := config/default.yaml
MODEL_DIR  := models/saved
LOG_DIR    := models/logs

help:
	@echo "GinkgoBrain - RL Trading Strategy Trainer"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "  install        安装依赖（创建虚拟环境并 pip install）"
	@echo "  test           运行 pytest 测试"
	@echo "  train-stock    训练股票策略（使用 config 中的 stock 配置）"
	@echo "  train-crypto   训练加密货币策略"
	@echo "  eval-stock     评估最新股票模型"
	@echo "  eval-crypto    评估最新加密货币模型"
	@echo "  tb             启动 TensorBoard"
	@echo "  clean          清理模型、日志、缓存文件"
	@echo ""
	@echo "可覆盖变量:"
	@echo "  CONFIG=...     配置文件路径（默认: $(CONFIG)）"
	@echo "  RUN=...        运行名称，传给 --run-name"
	@echo "  MODEL=...      模型路径，传给 --model（eval 时使用）"
	@echo "  EPISODES=...   评估 episode 数（默认: 5）"
	@echo ""
	@echo "示例:"
	@echo "  make train-crypto RUN=btc_sac CONFIG=config/sac.yaml"
	@echo "  make eval-crypto  MODEL=models/saved/BTCUSDT_ppo EPISODES=10"

# ── 环境 ────────────────────────────────────────────────────────────────────

install:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "Done. Activate with: source .venv/bin/activate"

# ── 测试 ────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

# ── 训练 ────────────────────────────────────────────────────────────────────

train-stock:
	$(PYTHON) train.py --mode stock --config $(CONFIG) $(if $(RUN),--run-name $(RUN),)

train-crypto:
	$(PYTHON) train.py --mode crypto --config $(CONFIG) $(if $(RUN),--run-name $(RUN),)

# ── 评估 ────────────────────────────────────────────────────────────────────

EPISODES ?= 5

eval-stock:
ifndef MODEL
	$(error 请指定 MODEL，例如: make eval-stock MODEL=models/saved/AAPL__ppo)
endif
	$(PYTHON) evaluate.py --mode stock --model $(MODEL) --config $(CONFIG) --episodes $(EPISODES)

eval-crypto:
ifndef MODEL
	$(error 请指定 MODEL，例如: make eval-crypto MODEL=models/saved/BTCUSDT_ppo)
endif
	$(PYTHON) evaluate.py --mode crypto --model $(MODEL) --config $(CONFIG) --episodes $(EPISODES)

# ── 工具 ────────────────────────────────────────────────────────────────────

tb:
	tensorboard --logdir $(LOG_DIR)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
	@echo "清理完成（模型文件保留，如需删除请手动执行 rm -rf $(MODEL_DIR) $(LOG_DIR)）"
