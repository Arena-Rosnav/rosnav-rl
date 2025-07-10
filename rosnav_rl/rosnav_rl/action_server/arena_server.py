from pathlib import Path
from time import sleep

import rospkg
from rl_utils.utils.constants import Simulator
from tools.states import get_arena_states

from rosnav_rl.observations import ObservationManager, get_required_observation_units
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.states import AgentStateContainer, SimulationStateContainer
from rosnav_rl.utils.utils import load_yaml

from .base_server import ActionServer, ObservationCollector


class ArenaActionServer(ActionServer):
    def _initialize_agent(self) -> RL_Agent:
        """
        Initializes and returns an RL_Agent instance.

        This method performs the following steps:
        1. Loads the training configuration from a YAML file.
        2. Initializes the simulation state container with parameters from the training configuration.
        3. Converts the simulation state container to an agent state container.
        4. Creates an RL_Agent instance with the agent configuration and agent state container.
        5. Loads the pre-trained model into the RL_Agent instance.

        Returns:
            RL_Agent: An instance of the RL_Agent class initialized with the loaded model and configuration.
        """
        import rl_utils.cfg as arena_cfg

        _rosnav_path = Path(rospkg.RosPack().get_path("rosnav_rl"))
        _model_dir = _rosnav_path / "agents" / self.agent_name

        training_cfg = arena_cfg.TrainingCfg.model_validate(
            load_yaml(_model_dir / "training_config.yaml")
        )
        self.simulation_state_container: SimulationStateContainer = get_arena_states(
            goal_radius=training_cfg.framework_cfg.general.goal_radius,
            max_steps=training_cfg.framework_cfg.general.max_num_moves_per_eps,
            is_discrete=training_cfg.agent_cfg.action_space.is_discrete,
            safety_distance=training_cfg.framework_cfg.general.safety_distance,
            robot_cfg=training_cfg.framework_cfg.robot,
            task_modules_cfg=training_cfg.framework_cfg.task,
        )
        agent_state_container: AgentStateContainer = (
            self.simulation_state_container.to_agent_state_container()
        )
        agent = RL_Agent(
            agent_cfg=training_cfg.agent_cfg,
            agent_state_container=agent_state_container,
        )
        agent.load_model(path=_model_dir / "best_model.zip")
        return agent

    def _initialize_observation_collector(self) -> ObservationCollector:
        """
        Initializes and returns an ObservationCollector instance.

        This method sets up an ObservationManager with the necessary parameters
        including namespace, observation structure, simulation state container,
        and other configurations.

        Returns:
            ObservationCollector: An instance of ObservationManager configured
            with the required observation units and simulation state container.
        """
        obs_manager = ObservationManager(
            ns=self.namespace,
            obs_structur=get_required_observation_units(
                self.agent.space_manager.observation_space_list
            ),
            simulation_state_container=self.simulation_state_container,
            wait_for_obs=False,
            is_single_env=True,
        )

        return obs_manager
