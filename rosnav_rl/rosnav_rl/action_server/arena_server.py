import yaml

from rosnav_rl.observations.factory.factory import (
    create_observation_manager_from_config,
)
from rosnav_rl.rl_agent import RL_Agent
from rosnav_rl.cfg.parameters import AgentParameters
from rosnav_rl.utils.agent_paths import (
    load_agent_spec,
    resolve_agent_dir,
    resolve_observations_config_path,
)

from .base_server import ActionServer, ObservationCollector


class ArenaActionServer(ActionServer):
    def _initialize_agent(self) -> RL_Agent:
        """Initialize and return an RL_Agent from a saved training config.

        Loads the ``AgentConfig`` from the training config and stores
        ``spec.parameters`` as ``agent_parameters`` for the observation pipeline.
        """
        model_dir = resolve_agent_dir(self.agent_name)
        self.logger.info(f"Loading agent from: {model_dir}")

        spec = load_agent_spec(model_dir)
        self._agent_spec = spec
        self.agent_parameters: AgentParameters = spec.parameters

        agent = RL_Agent(spec)
        agent.load_model(path=model_dir / "best_model.zip")
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

        obs_manager = create_observation_manager_from_config(
            config=config,
            node=self.node,
            ns=str(self.namespace),
            simulation_state_container=self.agent_parameters,
            wait_for_obs=False,
        )

        return obs_manager
