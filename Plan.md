# Quant Research Backtesting Framework – **PLAN.md**

> **AUTHOR:** ChatGPT‑o3 | **DATE:** 2025‑05‑17
> **VERSION:** 1.0
> **STATUS:** Draft – ready for review

---

## 1. Vision & Scope

Create a **modular, open‑source backtesting research platform** that supports Felix’s academic work (volatility modelling, ML‑driven factors) *and* serves as a professional showcase.

**Key goals**

1. **Research‑grade accuracy** – correct handling of corporate actions, survivorship, look‑ahead bias, realistic costs.
2. **Flexibility** – easy to plug in new signals (GARCH, transformers, etc.) and new asset classes.
3. **Reproducibility** – config‑driven experiments, version‑controlled data snapshots.
4. **Transparency & Learning** – code that is readable, documented, unit‑tested.
5. **Extensibility** – path to CLI, Jupyter widgets, Streamlit GUI, and live trading.

---

## 2. High‑Level Architecture

```text
┌────────────────────────────────────┐
│  1. Data Layer                     │ ← fetch, cache, describe
└────────────────────────────────────┘
          │ raw/dataframes
          ▼
┌────────────────────────────────────┐
│  2. Signal Layer                   │ ← factors, ML models
└────────────────────────────────────┘
          │ signals (α)
          ▼
┌────────────────────────────────────┐
│  3. Portfolio Layer                │ ← weights & constraints
└────────────────────────────────────┘
          │ weights
          ▼
┌────────────────────────────────────┐
│  4. Backtest Engine                │ ← execution + PnL
└────────────────────────────────────┘
          │ results
          ▼
┌────────────────────────────────────┐
│  5. Analytics & Reporting          │ ← metrics, plots, pdf
└────────────────────────────────────┘
```

Optional Interfaces → CLI, Notebook widgets, Streamlit dashboard.

---

## 3. Repository Directory Layout

```bash
quant-backtester/
├── README.md                # high‑level intro
├── PLAN.md                  # this file
├── requirements.txt         # pinned versions
├── .gitignore
├── data/                    # raw & interim data
│   ├── README.md            # dataset inventory & metadata schema
│   └── risk_free.csv        # example: 3‑Month T‑Bill
├── src/                     # package code (importable)
│   ├── __init__.py
│   ├── config.py            # global paths, constants
│   ├── data_loader.py
│   ├── metadata.py          # dataset registry helpers
│   ├── signals.py
│   ├── portfolio.py
│   ├── backtester.py
│   ├── analytics.py
│   └── utils.py
├── configs/                 # YAML/JSON experiment configs
│   └── example.yaml
├── notebooks/               # exploratory analyses / demos
│   └── demo.ipynb
├── tests/                   # pytest unit tests
│   └── test_backtester.py
└── docs/                    # optional MkDocs or Sphinx site
```

> **Tip:** treat `src/` as an installable package (`pip install -e .`) to enable clean imports like `from qb.signals import ema_signal`.

---

## 4. Development Roadmap (Milestones & Sub‑Tasks)

> **Legend:** ✅ = done  🔜 = next  🔲 = later
> **Phase 0** completed = repo scaffolding (README, PLAN, gitignore)

### **Phase 1 – Data Foundation** 🔜

| #   | Task                  | Details                                                                                                  |
| --- | --------------------- | -------------------------------------------------------------------------------------------------------- |
| 1.1 | **Metadata schema**   | YAML spec (`datasets.yaml`) with fields: `name`, `file`, `freq`, `source`, `description`, `last_updated` |
| 1.2 | **`data_loader.py`**  | Functions: `fetch_yfinance`, `load_csv`, caching, date parsing                                           |
| 1.3 | **Risk‑free series**  | Download 3M T‑Bill from FRED, save `risk_free.csv`, register in metadata                                 |
| 1.4 | **Dataset unit test** | pytest check: columns, NA, correct frequency                                                             |

### **Phase 2 – Minimal Viable Backtest Loop**

| #   | Task                | Depends | Details                                                |
| --- | ------------------- | ------- | ------------------------------------------------------ |
| 2.1 | **Simple signal**   | 1.x     | EWMA volatility or 50/200 SMA crossover                |
| 2.2 | **`portfolio.py`**  | 2.1     | Translate signal to {0,1} weights or vol‑target sizing |
| 2.3 | **`backtester.py`** | 2.2     | Daily rebalancing, fixed slippage & fees               |
| 2.4 | **Basic analytics** | 2.3     | CAGR, Sharpe, max DD, plot equity & DD                 |
| 2.5 | **Notebook demo**   | 2.x     | `notebooks/demo.ipynb` runs end‑to‑end                 |

### **Phase 3 – Configuration & CLI**

| #   | Task                                 | Details |
| --- | ------------------------------------ | ------- |
| 3.1 | YAML config schemas (`configs/`)     |         |
| 3.2 | `main.py` entrypoint (argparse)      |         |
| 3.3 | Parameter sweep helper (grid/random) |         |

### **Phase 4 – Robustness Layer**

| #   | Task                                                   | Details |
| --- | ------------------------------------------------------ | ------- |
| 4.1 | Transaction cost model (spread + fee)                  |         |
| 4.2 | Position sizing constraints (max weight, turnover cap) |         |
| 4.3 | Vectorised performance for speed (NumPy)               |         |
| 4.4 | Factor regression analytics (Fama‑French)              |         |

### **Phase 5 – ML & Advanced Features**

\| 5.x | Transformer / GARCH modules for volatility prediction |
\| 5.x | Hyper‑param optimisation (Optuna) |
\| 5.x | Parallel backtests (joblib / Ray) |

### **Phase 6 – UI & Reporting**

\| 6.1 | Streamlit dashboard (optional) |
\| 6.2 | PDF report generator (WeasyPrint or LaTeX) |

---

## 5. Coding Conventions & Tooling

| Aspect            | Choice                         |
| ----------------- | ------------------------------ |
| **Language**      | Python ≥3.11                   |
| **Style**         | PEP8 via *ruff* / *black*      |
| **Static Checks** | `mypy` for type hints          |
| **Testing**       | `pytest`, `coverage`           |
| **Linter**        | `ruff`                         |
| **Docs**          | Markdown, optionally MkDocs    |
| **CI**            | GitHub Actions (pytest + lint) |

---

## 6. Contribution Workflow

1. **Create feature branch** – `feature/phase2-signal-engine`
2. Commit with conventional commits (`feat: add sma_signal`)
3. Pull request → CI must pass
4. Review, squash & merge

---

## 7. Stretch Goals & Ideas Bank

* Live trading adapter (Interactive Brokers via `ib_insync`)
* Futures & options support (contract roll logic)
* Walk‑forward out‑of‑sample evaluation scaffold
* Research notes folder (`research/`) for academic write‑ups

---

## 8. Immediate Next Actions (Checklist)

* [ ] **Create `requirements.txt`** with core libs (`pandas`, `numpy`, `yfinance`, etc.)
* [ ] **Implement Phase 1 → metadata schema + loader**
* [ ] **Fetch risk‑free series, commit to `data/`**
* [ ] **Set up GitHub Actions CI template**

---

**Let’s build.**
