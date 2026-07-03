import yaml

from rosnav_rl.observations import ObservationManager
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.utils.agent_paths import (
    resolve_agent_dir,
    resolve_observations_config_path,
)

from .base_server import ActionServer, ObservationCollector


class ArenaActionServer(ActionServer):
    def _initialize_agent(self) -> RL_Agent:
        """Initialize and return an RL_Agent from a saved training config.

        Dispatches on the saved agent's framework (SB3 vs. DreamerV3) via
        ``RL_Agent.from_agent_dir``, and stores ``spec.parameters`` as
        ``agent_parameters`` for the observation pipeline.
        """
        model_dir = resolve_agent_dir(self.agent_name)
        self.logger.info(f"Loading agent from: {model_dir}")

        agent = RL_Agent.from_agent_dir(model_dir)
        self._agent_spec = agent.spec
        self.agent_parameters: AgentParameters = agent.spec.parameters
        return agent

    def _initialize_observation_collector(self) -> ObservationCollector:
        """
        Initializes and returns an ObservationCollector instance.

        Uses the ObservationManager.from_config() factory with the observations.yaml
        the agent was actually trained against, to set up ROS2 topic subscribers for
        collecting sensor data needed by the RL agent.

        Returns:
            ObservationCollector: Configured ObservationManager instance.
        """
        obs_config_path = resolve_observations_config_path(self._agent_spec)

        with open(obs_config_path, "r") as f:
            config = yaml.safe_load(f)

        obs_manager = ObservationManager.from_config(
            config=config,
            node=self.node,
            ns=str(self.namespace),
            simulation_state_container=self.agent_parameters,
            wait_for_obs=False,
        )

        return obs_manager
