class BaseSpaceEncoder:
    def __init__(
        self,
        laser_num_beams: int,
        laser_max_range: float,
        radius: float,
        holonomic: bool,
        actions: dict,
        action_space_discrete: bool,
        stacked: bool = False,
        *args,
        **kwargs
    ):
        self._laser_num_beams = laser_num_beams
        self._laser_max_range = laser_max_range
        self._radius = radius
        self._is_holonomic = holonomic
        self._actions = actions
        self._is_action_space_discrete = action_space_discrete
        self._stacked = stacked

    @property
    def observation_space(self):
        raise NotImplementedError()

    @property
    def action_space(self):
        raise NotImplementedError()

    def decode_action(self, action):
        raise NotImplementedError()

    def encode_observation(self):
        raise NotImplementedError()
