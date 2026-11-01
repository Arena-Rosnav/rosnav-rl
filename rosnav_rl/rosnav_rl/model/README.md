# Model Package

> Back to [README](../../README.md) · [Developer Guide](../../GUIDE.md) · [Tutorials](../../TUTORIALS.md)

The `model` package defines the RL model abstraction layer for **rosnav_rl**.
It decouples algorithm-specific details from the rest of the training pipeline
through a **Factory + Strategy** pattern, making it straightforward to swap
frameworks and algorithms without touching training or inference code.

## Directory Structure

```
model/
├── model.py                    # RL_Model – abstract base class
├── model_factory.py            # ModelFactory – registry-based factory
├── stable_baselines3/          # Stable Baselines 3 integration
│   ├── sb3_model.py            # StableBaselinesModel (RL_Model impl)
│   ├── cfg/                    # Pydantic configuration hierarchy
│   │   ├── base.py             # SBAlgorithmParameters → On/OffPolicyParameters
│   │   ├── framework.py        # StableBaselinesCfg (framework envelope)
│   │   ├── ppo.py              # PPO_Cfg / PPO_Algorithm_Cfg
│   │   ├── a2c.py              # A2C_Cfg / A2C_Algorithm_Cfg
│   │   ├── trpo.py             # TRPO_Cfg / TRPO_Algorithm_Cfg
│   │   ├── sac.py              # SAC_Cfg / SAC_Algorithm_Cfg
│   │   ├── td3.py              # TD3_Cfg / TD3_Algorithm_Cfg
│   │   ├── tqc.py              # TQC_Cfg / TQC_Algorithm_Cfg
│   │   ├── crossq.py           # CrossQ_Cfg / CrossQ_Algorithm_Cfg
│   │   ├── callbacks.py        # CallbacksCfg
│   │   ├── lr_schedule.py      # LearningRateSchedulerCfg
│   │   ├── normalization.py    # NormalizationCfg (VecNormalize)
│   │   └── transfer.py         # TransferWeightsCfg
│   └── policy/                 # Neural-network architecture descriptions
│       ├── agent_factory.py    # AgentFactory (architecture registry)
│       ├── base_policy.py      # StableBaselinesPolicyDescription (ABC)
│       ├── constants.py        # POLICY_TYPE map, BASE_AGENT_ATTR
│       ├── custom_policy.py    # Custom SB3 policy utilities
│       ├── feature_extractors/ # Custom feature-extractor classes
│       └── sb3_policy/         # Concrete architecture descriptions
└── dreamerv3/                  # DreamerV3 world-model integration
    ├── dreamerv3_model.py      # DreamerV3Model (RL_Model impl)
    ├── cfg.py                  # DreamerV3Cfg
    ├── dreamer.py              # Core Dreamer logic
    ├── models.py               # World-model networks
    ├── networks.py             # Neural-network building blocks
    └── ...
```

## Core Abstractions

### `RL_Model` (Abstract Base Class)

Defined in `model.py`, `RL_Model` is the contract every framework must fulfil:

| Method / Property         | Description                                        |
| ------------------------- | -------------------------------------------------- |
| `setup_model()`           | Initialise the underlying algorithm                |
| `train()`                 | Run a training iteration                           |
| `save()` / `load()`       | Persist & restore model weights                    |
| `get_action(obs)`         | Inference — return an action for an observation    |
| `transfer_weights()`      | Optional cross-model weight transfer               |
| `from_framework_cfg()`    | **Class method** — construct from a `FrameworkCfg` |
| `model`                   | Property — access the wrapped algorithm object     |
| `algorithm_cfg`           | Property — access the Pydantic config              |
| `observation_space_list`  | Property — observation spaces used by this model   |
| `stack_size`              | Property — temporal frame-stacking depth           |
| `parameter_number`        | Property — total trainable parameter count         |

### `ModelFactory`

Defined in `model_factory.py`, the factory maintains a `name → class` registry.
New frameworks are registered with a decorator or explicit call:

```python
@ModelFactory.register(SupportedRLFrameworks.STABLE_BASELINES3)
class StableBaselinesModel(RL_Model):
    ...
```

Creating a model instance from a plain dict → Pydantic config → model:

```python
from rosnav_rl.model.model_factory import ModelFactory

model = ModelFactory.create_model_instance(
    framework_cfg=stable_baselines_cfg,   # StableBaselinesCfg
    rl_agent=agent,
)
```

The factory delegates to `model_class.from_framework_cfg()` — no
framework-specific `if/elif` branching is required.

---

## Stable Baselines 3 Integration

### Supported Algorithms

| Algorithm       | Family     | Source          | Config Class             |
| --------------- | ---------- | -------------- | ------------------------ |
| [**PPO**](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)         | On-policy  | `stable_baselines3`  | `PPO_Cfg`          |
| [**A2C**](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)         | On-policy  | `stable_baselines3`  | `A2C_Cfg`          |
| [**TRPO**](https://sb3-contrib.readthedocs.io/en/master/modules/trpo.html)        | On-policy  | `sb3_contrib`        | `TRPO_Cfg`         |
| [**RecurrentPPO**](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)| On-policy  | `sb3_contrib`        | `PPO_Cfg` (LSTM arch) |
| [**SAC**](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)         | Off-policy | `stable_baselines3`  | `SAC_Cfg`          |
| [**TD3**](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html)         | Off-policy | `stable_baselines3`  | `TD3_Cfg`          |
| [**DDPG**](https://stable-baselines3.readthedocs.io/en/master/modules/ddpg.html)        | Off-policy | `stable_baselines3`  | (uses `SBAlgorithmCfg`) |
| [**TQC**](https://sb3-contrib.readthedocs.io/en/master/modules/tqc.html)         | Off-policy | `sb3_contrib`        | `TQC_Cfg`          |
| [**CrossQ**](https://sb3-contrib.readthedocs.io/en/master/modules/crossq.html)      | Off-policy | `sb3_contrib`        | `CrossQ_Cfg`       |

### Configuration Hierarchy

All configurations are **Pydantic v2 `BaseModel`** instances, enabling
`model_validate()` / `model_dump()` round-trip serialisation and automatic
validation.

```
SBAlgorithmParameters          (base — shared by every algorithm)
├── OnPolicyParameters         (on-policy family)
│   ├── PPO_Algorithm_Cfg
│   ├── A2C_Algorithm_Cfg
│   └── TRPO_Algorithm_Cfg
└── OffPolicyParameters        (off-policy family)
    ├── SAC_Algorithm_Cfg
    ├── TD3_Algorithm_Cfg
    ├── TQC_Algorithm_Cfg
    └── CrossQ_Algorithm_Cfg
```

**`SBAlgorithmParameters`** — universal fields:

- `total_timesteps`, `learning_rate`, `batch_size`, `gamma`, `device`, `seed`, …

**`OnPolicyParameters`** adds:

- `total_batch_size`, `n_epochs`, `gae_lambda`, `ent_coef`, `vf_coef`,
  `max_grad_norm`, `use_sde`, …

**`OffPolicyParameters`** adds:

- `buffer_size`, `learning_starts`, `tau`, `train_freq`, `gradient_steps`,
  `optimize_memory_usage`, `use_sde`, …

Each algorithm cfg class (e.g. `PPO_Algorithm_Cfg`) only declares the fields
**unique** to that algorithm; shared defaults are inherited.

### Envelope Pattern

Every algorithm parameters class is wrapped in an `SBAlgorithmCfg` envelope:

```python
class PPO_Cfg(SBAlgorithmCfg):
    parameters: PPO_Algorithm_Cfg
```

The envelope holds:
- `architecture_name` — registry key for the neural-network description
- `checkpoint` — model file to load (default `"last_model"`)
- `transfer_weights` — optional weight-transfer config
- `parameters` — algorithm-specific hyper-parameters
- `normalization` — optional `VecNormalize` settings
- `callbacks` — training callback config

### Framework Envelope

`StableBaselinesCfg` is the top-level config fed to `ModelFactory`. Its
`algorithm` field is a **discriminated union** of all supported `*_Cfg`
classes, so plain dicts are automatically parsed to the correct type:

```python
from rosnav_rl.model.stable_baselines3.cfg import StableBaselinesCfg

cfg = StableBaselinesCfg.model_validate({
    "name": "stable_baselines3",
    "algorithm": {
        "architecture_name": "AGENT_1",
        "parameters": {
            "algorithm_name": "PPO",
            "total_timesteps": 5_000_000,
            "clip_range": 0.15,
        }
    }
})
```

### Policy Descriptions

Each neural-network architecture is a concrete
`StableBaselinesPolicyDescription` registered with the `AgentFactory`:

```python
@AgentFactory.register("AGENT_1")
class AGENT_1(StableBaselinesPolicyDescription):
    algorithm_class = PPO
    observation_spaces = [
        spaces.perception.ReducedLaserScanSpace,
        spaces.navigation.DistAngleToSubgoalSpace,
        spaces.dynamics.LastActionSpace,
    ]
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=256)
    net_arch = dict(pi=[64, 64], vf=[64, 64])
    activation_fn = nn.ReLU
```

The `POLICY_TYPE` mapping in `constants.py` resolves each SB3 algorithm class
to its SB3 policy string (`"MultiInputPolicy"` or `"MultiInputLstmPolicy"`).

### Recurrent Models

`RecurrentPPO` (LSTM-based PPO) uses the **same** `StableBaselinesModel` class.
Recurrence is detected **structurally** at runtime:

```python
@property
def _is_recurrent(self) -> bool:
    return hasattr(self._model.policy, "lstm_actor")
```

No `isinstance(model, RecurrentPPO)` check is needed, making the code agnostic
to the concrete algorithm class.

---

## DreamerV3 Integration

The `dreamerv3/` sub-package implements a PyTorch-based DreamerV3 world-model
framework.  `DreamerV3Model` extends `RL_Model` and is registered with
`ModelFactory` under `SupportedRLFrameworks.DREAMER_V3`.

See [dreamerv3/package_description.md](dreamerv3/package_description.md) for
architecture details.

---

## Adding a New Algorithm

### Within Stable Baselines 3

1. **Create a config file** in `stable_baselines3/cfg/` (e.g. `myalgo.py`):

   ```python
   from .base import OnPolicyParameters, SBAlgorithmCfg  # or OffPolicyParameters

   class MyAlgo_Algorithm_Cfg(OnPolicyParameters):
       algorithm_name = "MyAlgo"
       my_param: float = 0.42

   class MyAlgo_Cfg(SBAlgorithmCfg):
       parameters: MyAlgo_Algorithm_Cfg
   ```

2. **Export** from `cfg/__init__.py`.

3. **Add to the union** in `cfg/framework.py`:

   ```python
   algorithm: Union[..., MyAlgo_Cfg, SBAlgorithmCfg]
   ```

4. **Register the policy type** in `policy/constants.py`:

   ```python
   from my_package import MyAlgo
   POLICY_TYPE[MyAlgo] = "MultiInputPolicy"
   ```

5. **Add to `_SupportedStableBaselinesModels`** in `utils/type_aliases/models.py`.

No changes to `sb3_model.py` or `model_factory.py` are necessary.

### Adding a New Framework

1. Subclass `RL_Model` and implement all abstract methods.
2. Implement `from_framework_cfg()` to extract your config.
3. Register with `ModelFactory`:

   ```python
   ModelFactory.register_model(SupportedRLFrameworks.MY_FRAMEWORK, MyModel)
   ```

4. Create a corresponding `FrameworkCfg` subclass.
