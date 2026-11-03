"""Flexible Curriculum base for ROS2-based RL training environments.

This module provides a reusable, framework-agnostic CurriculumBase that manages
curriculum stages (sets of environment/task parameters), communicates those
parameters to task generator nodes via ROS2 parameter services, and offers a
pluggable mechanism to check performance and advance or retreat stages.

Design goals
 - Accept several stage formats and normalize them to a list of stage dicts
 - Allow configurable parameter client discovery (node name template or explicit)
 - Provide hooks for lifecycle events (on_apply, on_advance, on_retreat)
 - Keep ROS2 parameter semantics: types, arrays, and service timeouts

Usage summary
 - Subclass CurriculumBase and implement `get_current_performance()` and
   `reset_performance_tracking()` using framework-specific metrics.
 - Instantiate with a ROS2 `Node`, a stages definition (see below), thresholds
   and number of environments.

Stage format (accepted)
 - Dict[str, List[Any]] where each key maps to a list of values per stage. E.g.
     {"n_static_obstacles": [[0,0],[2,4]], "goal_radius": [1.5, 1.2]}
 - List[Dict[str, Any]] where each list item is a stage mapping param->value. E.g.
     [{"n_static_obstacles": [0,0], "goal_radius":1.5}, {...}]

Methods of interest
 - get_curriculum_stages(): returns normalized list of stage dicts
 - advance_curriculum()/retreat_curriculum(): change stage and apply params
 - check_thresholds_and_update(): uses get_current_performance() to decide

Edge cases
 - Empty stages: initialization will raise ValueError
 - Partial parameter application: a stage may omit some parameters; previous
   values are not re-used unless included in the stage definition

Examples
 >>> stages = [{"n_static_obstacles": [0,0], "goal_radius":1.5}, {"n_static_obstacles":[2,4], "goal_radius":1.2}]
 >>> curriculum = CurriculumBase(node, stages, 'succ', 0.9, 0.3, num_envs=1)

"""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from arena_runtime_msgs.srv import QueryTaskModes, QueueEpisode
from diagnostic_msgs.msg import KeyValue
from rcl_interfaces.msg import Parameter, ParameterType
from rcl_interfaces.srv import SetParameters
from rclpy.client import Client
from rclpy.node import Node

StageInput = dict[str, list[Any]] | list[dict[str, Any]]


class CurriculumBase(ABC):
    """Abstract, flexible curriculum base class.

    Args:
        node: rclpy Node used to create clients and spin for service responses
        train_stages: stages definition (see module docstring)
        threshold_type: metric type used for thresholds (user-specific string)
        upper_threshold: value to advance stage
        lower_threshold: value to retreat stage
        num_envs: number of task-generator nodes to configure
        parameter_node_template: template for task generator node name, with an
            optional `{i}` placeholder for env index. Default is the registry node
            path '/arena/env_{i}/arena_env_node'
        parameter_service_name: suffix for service (default 'set_parameters')
        starting_stage: initial curriculum stage index (default 0)
        verbose: verbosity level (0 quiet, >0 prints/logs)
        timeout: service call timeout in seconds
        modes: axis -> mode names queued before the initial stage is applied

    Hooks (callable signature): on_apply(stage_index, stage_dict), on_advance(idx), on_retreat(idx)
    """

    def __init__(
        self,
        node: Node,
        train_stages: StageInput,
        threshold_type: str,
        upper_threshold: float,
        lower_threshold: float,
        num_envs: int,
        parameter_node_template: str | None = None,
        parameter_service_name: str = "set_parameters",
        starting_stage: int = 0,
        verbose: int = 0,
        timeout: float = 10.0,
        modes: dict[str, str | list[str]] | None = None,
    ):
        self.node = node
        self.threshold_type = threshold_type
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.num_envs = max(1, int(num_envs))
        self.parameter_node_template = (
            parameter_node_template if parameter_node_template is not None else "/arena/env_{i}/arena_env_node"
        )
        self.parameter_service_name = parameter_service_name
        self.verbose = verbose
        self.timeout = timeout

        # Normalized stages: list of dicts {param_name: value}
        self._stages: list[dict[str, Any]] = self._normalize_train_stages(train_stages)
        if not self._stages:
            raise ValueError("No curriculum stages provided")

        # Validate and set starting stage
        self.starting_stage = max(0, min(starting_stage, len(self._stages) - 1))
        self.curriculum_index = self.starting_stage
        self.max_index = len(self._stages)

        # Parameter clients keyed by node_name
        self.parameter_clients = self._init_parameter_clients()
        self.task_mode_clients = self._init_task_mode_clients()
        self._namespace_clients = {
            name: self.node.create_client(QueryTaskModes, f"{name}/query/task_modes") for name in self.task_mode_clients
        }
        self._mode_namespaces: dict[str, dict[str, str]] = {}

        # Lifecycle hooks
        self._on_apply: list[Callable[[int, dict[str, Any]], None]] = []
        self._on_advance: list[Callable[[int], None]] = []
        self._on_retreat: list[Callable[[int], None]] = []

        # Nodes confirmed to lack config/queue_episode; task-prefixed params fall
        # back to SetParameters for these (same as legacy behaviour).
        self._non_arena_env_nodes: set[str] = set()

        if modes:
            self._queue_episode(modes)

        # Apply initial stage
        self._apply_curriculum()

    # ------------------------- Stage normalization -------------------------
    def _normalize_train_stages(self, train_stages: StageInput) -> list[dict[str, Any]]:
        """Convert accepted stage formats into a list of stage dicts.

        Accepts two formats:
        - dict[str, list]: each key maps to a list of per-stage values; all lists
          must have the same length. This will be transposed into a list of
          stage dictionaries.
        - list[dict]: already a list of stage dictionaries; each item must be a dict.

        Returns:
            A list of stage dictionaries in canonical form.

        Raises:
            ValueError: if lists have mismatched lengths or a list item is not a dict.
            TypeError: if the input type is not supported.
        """
        if isinstance(train_stages, dict):
            # ensure all values are lists and have same length
            keys = list(train_stages.keys())
            lengths = [len(v) for v in train_stages.values()]
            if not lengths or len(set(lengths)) != 1:
                raise ValueError("All parameter lists must be non-empty and the same length")
            n_stages = lengths[0]
            stages: list[dict[str, Any]] = []
            for i in range(n_stages):
                stage = {k: train_stages[k][i] for k in keys}
                stages.append(stage)
            return stages

        if isinstance(train_stages, list):
            # shallow copy, validate that each item is a dict
            for i, s in enumerate(train_stages):
                if not isinstance(s, dict):
                    raise ValueError(f"Stage at index {i} is not a dict")
            return [dict(s) for s in train_stages]

        raise TypeError("train_stages must be dict or list of dicts")

    # ------------------------- Parameter client handling -------------------------
    def _init_parameter_clients(self) -> dict[str, Client]:
        """Create ROS2 service clients for each task-generator node.

        The method uses `self.parameter_node_template` to format node names for
        each environment index. It returns a mapping node_name -> client
        (rclpy client object). Clients are created but not checked for availability
        here; callers should use `wait_for_service` with a timeout per-call.

        Returns:
            Dict[node_name, client]
        """
        clients: dict[str, Client] = {}
        for i in range(self.num_envs):
            if "{i}" in self.parameter_node_template:
                node_name = self.parameter_node_template.format(i=i)
            else:
                node_name = self.parameter_node_template
            service_name = f"{node_name}/{self.parameter_service_name}"
            clients[node_name] = self.node.create_client(SetParameters, service_name)
            if self.verbose > 0:
                print(f"[CURRICULUM_BASE] Created parameter client for {node_name} (service: {service_name})")

        if self.verbose > 0:
            print(f"[CURRICULUM_BASE] Created {len(clients)} parameter clients total")
        return clients

    def _init_task_mode_clients(self) -> dict[str, Client]:
        """Create ROS2 service clients for config/queue_episode on each node."""
        clients: dict[str, Client] = {}
        for i in range(self.num_envs):
            if "{i}" in self.parameter_node_template:
                node_name = self.parameter_node_template.format(i=i)
            else:
                node_name = self.parameter_node_template
            service_name = f"{node_name}/config/queue_episode"
            clients[node_name] = self.node.create_client(QueueEpisode, service_name)
        return clients

    # ------------------------- Hooks API -------------------------
    def register_on_apply(self, fn: Callable[[int, dict[str, Any]], None]) -> None:
        """Register a callback invoked after a stage is applied.

        The callback signature must be (stage_index: int, stage_dict: Dict[str, Any]).
        Callbacks are called even if applying parameters failed, allowing observers
        to record the attempted stage or take corrective action.
        """
        self._on_apply.append(fn)

    def register_on_advance(self, fn: Callable[[int], None]) -> None:
        """Register a callback invoked when the curriculum advances.

        The callback receives the new stage index as single integer argument.
        """
        self._on_advance.append(fn)

    def register_on_retreat(self, fn: Callable[[int], None]) -> None:
        """Register a callback invoked when the curriculum retreats.

        The callback receives the new stage index as single integer argument.
        """
        self._on_retreat.append(fn)

    def _call_hooks(self, hooks: list[Callable[..., None]], *args: Any, **kwargs: Any):
        """Safely call registered hooks.

        Any exception raised by a hook is caught and logged (when verbose>0).
        Hooks are executed in registration order.
        """
        for h in hooks:
            try:
                h(*args, **kwargs)
            except Exception:
                if self.verbose > 0:
                    print("Ignoring exception in hook", h)

    # ------------------------- Parameter conversion and setting -------------------------
    def _param_to_rcl_param(self, name: str, value: object) -> Parameter:
        param = Parameter()
        param.name = name
        if isinstance(value, int):
            param.value.type = ParameterType.PARAMETER_INTEGER
            param.value.integer_value = int(value)
        elif isinstance(value, float):
            param.value.type = ParameterType.PARAMETER_DOUBLE
            param.value.double_value = float(value)
        elif isinstance(value, str):
            param.value.type = ParameterType.PARAMETER_STRING
            param.value.string_value = value
        elif isinstance(value, bool):
            param.value.type = ParameterType.PARAMETER_BOOL
            param.value.bool_value = value
        elif isinstance(value, list):
            # Decide array type by uniform element types
            if all(isinstance(x, int) for x in value):
                param.value.type = ParameterType.PARAMETER_INTEGER_ARRAY
                param.value.integer_array_value = value
            elif all(isinstance(x, float) for x in value):
                param.value.type = ParameterType.PARAMETER_DOUBLE_ARRAY
                param.value.double_array_value = value
            elif all(isinstance(x, str) for x in value):
                param.value.type = ParameterType.PARAMETER_STRING_ARRAY
                param.value.string_array_value = value
            elif all(isinstance(x, bool) for x in value):
                param.value.type = ParameterType.PARAMETER_BOOL_ARRAY
                param.value.bool_array_value = value
            elif all(isinstance(x, bytes) for x in value):
                param.value.type = ParameterType.PARAMETER_BYTE_ARRAY
                param.value.byte_array_value = value
            else:
                raise TypeError(f"Unsupported array element types for parameter {name}")
        else:
            raise TypeError(f"Unsupported parameter type for {name}: {type(value)}")
        return param

    def _split_task_params(
        self, param_dict: dict[str, Any], namespaces: dict[str, str]
    ) -> tuple[list[Parameter], dict[str, Any]]:
        """Partition param_dict into (mode params by full name, plain_dict).

        Keys of the form ``task.<mode>.<leaf>`` go to the mode params under the
        env's namespace of ``<mode>``, ``task.episode.*`` node parameters go to plain_dict.
        """
        task: list[Parameter] = []
        plain: dict[str, Any] = {}

        for key, val in param_dict.items():
            parts = key.split(".", 2)
            if len(parts) == 3 and parts[0] == "task" and parts[1] != "episode":
                if isinstance(val, list) and not val:
                    continue
                try:
                    task.append(
                        self._param_to_rcl_param(
                            f"{namespaces[parts[1]]}.{parts[2]}" if parts[1] in namespaces else key, val
                        )
                    )
                except TypeError as exc:
                    if self.verbose > 0:
                        print(f"[CURRICULUM_BASE] Failed to convert {key}={val}: {exc}")
                    plain[key] = val
            else:
                plain[key] = val

        return task, plain

    def _send_set_parameters(self, node_name: str, params: list[Parameter]) -> bool:
        """Send a SetParameters request for the given pre-built param list."""
        client = self.parameter_clients.get(node_name)
        if client is None:
            if self.verbose > 0:
                print(f"[CURRICULUM_BASE] No client found for node {node_name}")
            return False

        if not client.wait_for_service(timeout_sec=2.0):
            if self.verbose > 0:
                print(f"[CURRICULUM_BASE] Service not available for node {node_name} after 2.0s timeout")
            return False

        request = SetParameters.Request(parameters=params)
        try:
            future = client.call_async(request)
            start = time.time()
            poll_timeout = min(self.timeout, 5.0)

            # Do not call rclpy.spin_once here: the node may already be spinning in
            # the application. Instead, poll the future with a short sleep to avoid
            # blocking the caller's executor. This is compatible with a background
            # spinning node.
            while not future.done():
                time.sleep(0.01)
                elapsed = time.time() - start
                if elapsed > poll_timeout:
                    if self.verbose > 0:
                        print(
                            f"[CURRICULUM_BASE] Timeout after {poll_timeout}s waiting for "
                            f"parameter response from {node_name}"
                        )
                    return False

            response = future.result()

            if response and all(r.successful for r in response.results):
                if self.verbose > 0:
                    print(f"[CURRICULUM_BASE] Successfully set {len(params)} parameters for {node_name}")
                return True

            if self.verbose > 0:
                print(f"[CURRICULUM_BASE] Parameter setting failed for {node_name}:")
                if response:
                    for i, result in enumerate(response.results):
                        if not result.successful:
                            param_name = params[i].name if i < len(params) else "unknown"
                            print(f"  - Parameter '{param_name}': {result.reason}")
                else:
                    print("  - No response received from service")
            return False
        except Exception as e:
            if self.verbose > 0:
                print(f"[CURRICULUM_BASE] Exception while setting parameters for {node_name}: {e}")
            return False

    def _set_parameters_batch(self, node_name: str, param_dict: dict[str, Any]) -> bool:
        """Set parameters for a single node with detailed error reporting.

        Keys matching ``task.<mode>.<leaf>`` are routed through QueueEpisode
        (buffered server-side, applied at next reset) for nodes that expose
        ``config/queue_episode``.  If the service is absent the first time it
        is checked, the node is marked as non-task-generator for this run and
        those keys fall back to the legacy SetParameters path.

        Args:
            node_name: Name of the target node
            param_dict: Dictionary of parameter name -> value

        Returns:
            True if all parameters were set successfully, False otherwise
        """
        task_params, plain_dict = self._split_task_params(param_dict, self._namespaces_of(node_name))

        ok = True

        if task_params and node_name not in self._non_arena_env_nodes:
            tm_client = self.task_mode_clients.get(node_name)
            if tm_client is not None and tm_client.wait_for_service(timeout_sec=2.0):
                req = QueueEpisode.Request()
                req.params = task_params
                ok = self._call_queue_episode(tm_client, node_name, req)
            else:
                # Service absent: mark node so we don't re-check and fall back.
                self._non_arena_env_nodes.add(node_name)
                if self.verbose > 0:
                    print(
                        f"[CURRICULUM_BASE] config/queue_episode not available for {node_name};"
                        " falling back to SetParameters for task-prefixed keys"
                    )
                ok = self._send_set_parameters(node_name, task_params) and ok

        elif task_params:
            # Already confirmed non-task-generator; send via SetParameters.
            ok = self._send_set_parameters(node_name, task_params) and ok

        # Plain (non-task-prefixed) parameters always go through SetParameters.
        if plain_dict:
            params: list[Parameter] = []
            for pname, pval in plain_dict.items():
                if isinstance(pval, list) and not pval:
                    continue
                try:
                    params.append(self._param_to_rcl_param(pname, pval))
                except Exception as e:
                    if self.verbose > 0:
                        print(f"[CURRICULUM_BASE] Failed to convert parameter {pname}={pval} for node {node_name}: {e}")
                    return False
            if params:
                ok = self._send_set_parameters(node_name, params) and ok

        if not plain_dict and not task_params:
            if self.verbose > 0:
                print(f"[CURRICULUM_BASE] No parameters to set for node {node_name}")
        return ok

    def _set_parameters(self, param_dict: dict[str, Any]) -> bool:
        if not self.parameter_clients:
            if self.verbose > 0:
                print("[CURRICULUM_BASE] No parameter clients available")
            return False

        success = True
        failed_nodes = []
        successful_nodes = []

        for node_name in list(self.parameter_clients.keys()):
            ok = self._set_parameters_batch(node_name, param_dict)
            if not ok:
                success = False
                failed_nodes.append(node_name)
                if self.verbose > 0:
                    print(f"[CURRICULUM_BASE] Failed to set parameters for {node_name}")
            else:
                successful_nodes.append(node_name)

        if self.verbose > 0:
            if successful_nodes:
                print(f"[CURRICULUM_BASE] Successfully set parameters for: {successful_nodes}")
            if failed_nodes:
                print(f"[CURRICULUM_BASE] Failed to set parameters for: {failed_nodes}")
            print(f"[CURRICULUM_BASE] Parameter setting complete: {success}")
        return success

    # ------------------------- Task-mode routing -------------------------
    _MODES_KEY = "modes"

    def _call(
        self, client: Client, req: QueryTaskModes.Request | QueueEpisode.Request, node_name: str, what: str
    ) -> QueryTaskModes.Response | QueueEpisode.Response | None:
        try:
            future = client.call_async(req)
            start = time.time()
            while not future.done():
                time.sleep(0.01)
                if time.time() - start > min(self.timeout, 5.0):
                    if self.verbose > 0:
                        print(f"[CURRICULUM_BASE] Timeout waiting for {what} on {node_name}")
                    return None
            return future.result()
        except Exception as e:
            if self.verbose > 0:
                print(f"[CURRICULUM_BASE] Exception in {what} for {node_name}: {e}")
            return None

    def _call_queue_episode(self, client: Client, node_name: str, req: QueueEpisode.Request) -> bool:
        response = self._call(client, req, node_name, "queue_episode")
        if response and response.success:
            return True
        if self.verbose > 0 and response is not None:
            print(f"[CURRICULUM_BASE] queue_episode rejected for {node_name}: {response.error_msg}")
        return False

    def _namespaces_of(self, node_name: str) -> dict[str, str]:
        """Mode name -> param namespace as the env reports it, asked once per node."""
        if node_name not in self._mode_namespaces:
            client = self._namespace_clients[node_name]
            response = (
                self._call(client, QueryTaskModes.Request(), node_name, "query_task_modes")
                if client.wait_for_service(timeout_sec=2.0)
                else None
            )
            pairs = response.namespaces if response else []
            self._mode_namespaces[node_name] = {pair.key.partition("/")[2]: pair.value for pair in pairs}
        return self._mode_namespaces[node_name]

    def _queue_episode_batch(self, node_name: str, modes: dict[str, str | list[str]]) -> bool:
        """Route the mode selection to config/queue_episode on a single node."""
        client = self.task_mode_clients.get(node_name)
        if client is None:
            return False
        if not client.wait_for_service(timeout_sec=2.0):
            if self.verbose > 0:
                print(f"[CURRICULUM_BASE] config/queue_episode not available for {node_name}")
            return False

        req = QueueEpisode.Request()
        req.modes = [
            KeyValue(key=axis, value=names if isinstance(names, str) else ",".join(names))
            for axis, names in modes.items()
        ]
        return self._call_queue_episode(client, node_name, req)

    def _queue_episode(self, modes: dict[str, str | list[str]]) -> bool:
        success = True
        for node_name in list(self.task_mode_clients.keys()):
            if not self._queue_episode_batch(node_name, modes):
                success = False
        return success

    # ------------------------- Curriculum application and control -------------------------
    def _apply_curriculum(self) -> bool:
        """Apply the current curriculum stage with enhanced debugging.

        Returns:
            True if parameters were applied successfully, False otherwise
        """
        stage = self._stages[self.curriculum_index]
        if self.verbose > 0:
            print(f"[CURRICULUM_BASE] Applying stage {self.curriculum_index}: {stage}")
            print(f"[CURRICULUM_BASE] Available parameter clients: {list(self.parameter_clients.keys())}")
            print(f"[CURRICULUM_BASE] Parameter node template: {self.parameter_node_template}")

        modes = stage.get(self._MODES_KEY)
        param_dict = {k: v for k, v in stage.items() if k not in (self._MODES_KEY, "train_mode")}

        ok = True
        if param_dict:
            ok = self._set_parameters(param_dict) and ok
        if modes:
            ok = self._queue_episode(modes) and ok
        # call hooks even if setting fails (observer may want to react)
        self._call_hooks(self._on_apply, self.curriculum_index, stage)
        return ok

    def advance_curriculum(self) -> bool:
        if self.curriculum_index < self.max_index - 1:
            self.curriculum_index += 1
            self._call_hooks(self._on_advance, self.curriculum_index)
            if self.verbose > 0:
                print(f"[CURRICULUM_BASE] Advanced to stage {self.curriculum_index}")
            return self._apply_curriculum()
        return True

    def retreat_curriculum(self) -> bool:
        if self.curriculum_index > 0:
            self.curriculum_index -= 1
            self._call_hooks(self._on_retreat, self.curriculum_index)
            if self.verbose > 0:
                print(f"[CURRICULUM_BASE] Retreated to stage {self.curriculum_index}")
            return self._apply_curriculum()
        return True

    def check_thresholds_and_update(self) -> bool:
        perf = self.get_current_performance()
        if perf is None:
            return True

        if perf >= self.upper_threshold:
            ok = self.advance_curriculum()
            self.reset_performance_tracking()
            return ok
        if perf <= self.lower_threshold:
            return self.retreat_curriculum()
        return True

    # ------------------------- Abstract methods to implement -------------------------
    @abstractmethod
    def get_current_performance(self) -> float | None:
        """Return the current performance value used for thresholds.

        Return None when no reliable metric is available yet.
        """
        raise NotImplementedError()

    @abstractmethod
    def reset_performance_tracking(self) -> None:
        """Reset any internal performance tracking when a stage changes.
        Implementations should clear counters or best-value records as needed.
        """
        raise NotImplementedError()

    # ------------------------- Convenience properties -------------------------
    @property
    def current_stage(self) -> int:
        return self.curriculum_index

    @property
    def is_final_stage(self) -> bool:
        return self.curriculum_index >= self.max_index - 1

    @property
    def is_first_stage(self) -> bool:
        return self.curriculum_index == 0

    def get_curriculum_stages(self) -> list[dict[str, Any]]:
        """Return the normalized list of stages (stage dicts)."""
        return list(self._stages)
