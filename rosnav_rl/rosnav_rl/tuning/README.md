# `rosnav_rl.tuning` — Optuna Hyperparameter Tuning

Framework-agnostic hyperparameter search for arena-rosnav RL agents, built on
[Optuna](https://optuna.readthedocs.io).  Any field in a `TrainingCfg` YAML can
be tuned without writing boilerplate — just describe the search space with
dot-notation paths.

Pruning is handled by the same **dual-inheritance pattern** as curriculum
learning (`CurriculumBase` / `StagedTrainCallback`): a shared abstract base
class owns the Optuna logic, and thin framework adapters bridge it into each
training loop.

---

## Module layout

```
rosnav_rl/tuning/
├── __init__.py           public API
├── cfg.py                TuningCfg — Pydantic model for the tuning YAML
├── search_space.py       FloatParam / IntParam / CategoricalParam
├── sampler.py            suggest_params(), apply_params()
├── pruner_base.py        TrialPrunerBase — abstract, framework-agnostic
├── sb3_pruner.py         SB3TrialPruner   — extends TrialPrunerBase + BaseCallback
├── dreamerv3_pruner.py   DreamerV3TrialPruner — extends TrialPrunerBase
└── callbacks.py          compatibility shim (deprecated, imports from sb3_pruner)
```

### Class hierarchy

```
TrialPrunerBase (ABC)
├── SB3TrialPruner       ← also extends SB3 BaseCallback
└── DreamerV3TrialPruner ← receives metrics via after_eval_hook()
```

This mirrors the curriculum pattern:

```
CurriculumBase (ABC)
├── StagedTrainCallback  ← SB3 integration
└── DreamerV3Curriculum  ← DreamerV3 integration
```

---

## Quick start

### 1. Write a tuning config YAML

```yaml
# ppo_tuning.yaml

base_config: sb_training_config.yaml   # relative to this file or absolute

study_name:       ppo_lr_search
n_trials:         40
direction:        maximize
metric:           mean_reward           # SB3: ep_rew_mean alias
trial_timesteps:  500_000               # shorten each trial to 500k steps
storage:          sqlite:///tuning.db   # persist across restarts (optional)

pruner:
  type:             median
  n_startup_trials: 5
  n_warmup_steps:   10

search_space:
  agent_cfg.framework.algorithm.parameters.learning_rate:
    type: float
    low:  1.0e-5
    high: 1.0e-3
    log:  true

  agent_cfg.framework.algorithm.parameters.n_steps:
    type: int
    low:  128
    high: 4096
    step: 128

  agent_cfg.framework.algorithm.parameters.gamma:
    type: float
    low:  0.90
    high: 0.9999

  agent_cfg.framework.algorithm.parameters.batch_size:
    type: categorical
    choices: [64, 128, 256, 512]
```

### 2. Run the tuning script

```bash
source ~/arena5_ws/install/setup.bash
python3 arena_training/scripts/tune_agent.py --config ppo_tuning.yaml
```

Results are saved to `ppo_lr_search_best_params.yaml` beside the config file.

---

## DreamerV3 tuning

For DreamerV3, the metric arrives via the `after_eval_fn` hook — replace the
`metric` field with `eval_return` and point at a DreamerV3 base config:

```yaml
# dreamer_tuning.yaml

base_config: dreamer_training_config.yaml

study_name:  dreamer_batch_search
n_trials:    20
direction:   maximize
metric:      eval_return

pruner:
  type: median
  n_startup_trials: 3
  n_warmup_steps:   5

search_space:
  agent_cfg.framework.training.batch_size:
    type: categorical
    choices: [8, 16, 32]

  agent_cfg.framework.training.batch_length:
    type: int
    low:  16
    high: 64
    step: 16

  agent_cfg.framework.world_model.model_lr:
    type: float
    low:  1.0e-5
    high: 1.0e-3
    log:  true
```

---

## Python API

### Pruning in custom training code

**SB3**

```python
import optuna
from rosnav_rl.tuning.sb3_pruner import SB3TrialPruner

def objective(trial: optuna.Trial) -> float:
    pruner = SB3TrialPruner(trial, metric="mean_reward", verbose=1)
    model.learn(total_timesteps=500_000, callback=[eval_cb, pruner])
    return pruner.best_metric or 0.0
```

**DreamerV3**

```python
import optuna
from rosnav_rl.tuning.dreamerv3_pruner import DreamerV3TrialPruner

def objective(trial: optuna.Trial) -> float:
    pruner = DreamerV3TrialPruner(trial, verbose=1)
    model.train(
        train_envs=train_envs,
        eval_envs=eval_envs,
        after_eval_fn=pruner.after_eval_hook,   # ← inject here
    )
    return pruner.best_metric or 0.0
```

### Extending to a new framework

Subclass `TrialPrunerBase` and implement exactly one method:

```python
from rosnav_rl.tuning.pruner_base import TrialPrunerBase

class MyFrameworkPruner(TrialPrunerBase):
    """Pruner for MyAwesomeRL."""

    def read_metric(self) -> float | None:
        """Pull the latest metric from the framework internals."""
        value = self._my_framework_logger.get("episode_reward")
        return float(value) if value is not None else None
```

Call `check_and_report()` at the appropriate hook point in your training loop:

```python
pruner = MyFrameworkPruner(trial, metric="episode_reward")

for epoch in range(n_epochs):
    train_one_epoch(...)
    pruner.check_and_report()   # reports to Optuna; raises TrialPruned if needed
```

---

## `TrialPrunerBase` reference

| Method | Description |
|---|---|
| `read_metric()` | **Abstract.** Return current metric or `None` (no data yet). |
| `report_metric(value)` | Report `value` to Optuna, update `best_metric`, raise `TrialPruned` if the study pruner decides. |
| `check_and_report()` | Call `read_metric()` then `report_metric()`. No-op if `None`. |
| `best_metric` | Best value reported so far. Use as the trial's return value. |

---

## Pruner types (Optuna)

Configure via `pruner.type` in the YAML:

| Type | Description |
|---|---|
| `median` (default) | Prune if below the median of completed trials at the same step. |
| `hyperband` | Multi-fidelity successive halving. |
| `percentile` | Keep only the top *N* % of trials. |
| `none` | Disable pruning (run all trials to completion). |

---

## Resuming a study

Set `storage: sqlite:///tuning.db` (or any Optuna-compatible URL) and the
study's history is preserved.  Restart with the same config and it will pick
up from where it left off:

```bash
python3 tune_agent.py --config ppo_tuning.yaml
# Ctrl-C ...
python3 tune_agent.py --config ppo_tuning.yaml   # continues from trial N+1
```

---

## Visualising results (Optuna Dashboard)

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///tuning.db
```

Open `http://localhost:8080` in your browser.
