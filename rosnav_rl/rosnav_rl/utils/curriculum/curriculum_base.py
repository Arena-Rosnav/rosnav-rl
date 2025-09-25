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

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Union
import time
from rclpy.node import Node
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterType


StageInput = Union[Dict[str, List[Any]], List[Dict[str, Any]]]


class CurriculumBase(ABC):
    """Abstract, flexible curriculum base class.

    Args:
        node: rclpy Node used to create clients and spin for service responses
        train_stages: stages definition (see module docstring)
        threshold_type: metric type used for thresholds (user-specific string)
        upper_threshold: value to advance stage
        lower_threshold: value to retreat stage
        num_envs: number of task-generator nodes to configure
        parameter_node_template: template for task generator node name, must contain
            one `{i}` placeholder for env index when `num_envs>1`. Default replicates
            existing behaviour: '/task_generator_node' or '/task_generator_node_{i}'
        parameter_service_name: suffix for service (default 'set_parameters')
        starting_stage: initial curriculum stage index (default 0)
        verbose: verbosity level (0 quiet, >0 prints/logs)
        timeout: service call timeout in seconds

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
        parameter_node_template: Optional[str] = None,
        parameter_service_name: str = "set_parameters",
        starting_stage: int = 0,
        verbose: int = 0,
        timeout: float = 10.0,
    ):
        self.node = node
        self.threshold_type = threshold_type
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.num_envs = max(1, int(num_envs))
        self.parameter_node_template = (
            parameter_node_template
            if parameter_node_template is not None
            else (
                "/task_generator_node_{i}"
                if self.num_envs > 1
                else "/task_generator_node"
            )
        )
        self.parameter_service_name = parameter_service_name
        self.verbose = verbose
        self.timeout = timeout

        # Normalized stages: list of dicts {param_name: value}
        self._stages: List[Dict[str, Any]] = self._normalize_train_stages(train_stages)
        if not self._stages:
            raise ValueError("No curriculum stages provided")

        # Validate and set starting stage
        self.starting_stage = max(0, min(starting_stage, len(self._stages) - 1))
        self.curriculum_index = self.starting_stage
        self.max_index = len(self._stages)

        # Parameter clients keyed by node_name
        self.parameter_clients = self._init_parameter_clients()

        # Lifecycle hooks
        self._on_apply: List[Callable[[int, Dict[str, Any]], None]] = []
        self._on_advance: List[Callable[[int], None]] = []
        self._on_retreat: List[Callable[[int], None]] = []

        # Apply initial stage
        self._apply_curriculum()

    # ------------------------- Stage normalization -------------------------
    def _normalize_train_stages(self, train_stages: StageInput) -> List[Dict[str, Any]]:
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
                raise ValueError(
                    "All parameter lists must be non-empty and the same length"
                )
            n_stages = lengths[0]
            stages: List[Dict[str, Any]] = []
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
    def _init_parameter_clients(self) -> Dict[str, Any]:
        """Create ROS2 service clients for each task-generator node.

        The method uses `self.parameter_node_template` to format node names for
        each environment index. It returns a mapping node_name -> client
        (rclpy client object). Clients are created but not checked for availability
        here; callers should use `wait_for_service` with a timeout per-call.

        Returns:
            Dict[node_name, client]
        """
        clients: Dict[str, Any] = {}
        for i in range(self.num_envs):
            if "{i}" in self.parameter_node_template:
                node_name = self.parameter_node_template.format(i=i)
            else:
                node_name = self.parameter_node_template
            service_name = f"{node_name}/{self.parameter_service_name}"
            clients[node_name] = self.node.create_client(SetParameters, service_name)
            if self.verbose > 0:
                print(
                    f"[CURRICULUM_BASE] Created parameter client for {node_name} (service: {service_name})"
                )

        if self.verbose > 0:
            print(f"[CURRICULUM_BASE] Created {len(clients)} parameter clients total")
        return clients

    # ------------------------- Hooks API -------------------------
    def register_on_apply(self, fn: Callable[[int, Dict[str, Any]], None]) -> None:
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

    def _call_hooks(self, hooks: List[Callable], *args, **kwargs):
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
    def _param_to_rcl_param(self, name: str, value: Any) -> Parameter:
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

    def _set_parameters_batch(self, node_name: str, param_dict: Dict[str, Any]) -> bool:
        """Set parameters for a single node with detailed error reporting.

        Args:
            node_name: Name of the target node
            param_dict: Dictionary of parameter name -> value

        Returns:
            True if all parameters were set successfully, False otherwise
        """
        client = self.parameter_clients.get(node_name)
        if client is None:
            if self.verbose > 0:
                print(f"[CURRICULUM_BASE] No client found for node {node_name}")
            return False

        # Test service availability with shorter timeout first
        if not client.wait_for_service(timeout_sec=2.0):
            if self.verbose > 0:
                print(
                    f"[CURRICULUM_BASE] Service not available for node {node_name} after 2.0s timeout"
                )
            return False

        params: List[Parameter] = []

        for pname, pval in param_dict.items():
            # Skip empty lists (explicitly absent)
            if isinstance(pval, list) and not pval:
                continue
            try:
                converted_param = self._param_to_rcl_param(pname, pval)
                params.append(converted_param)
            except Exception as e:
                if self.verbose > 0:
                    print(
                        f"[CURRICULUM_BASE] Failed to convert parameter {pname}={pval} for node {node_name}: {e}"
                    )
                return False

        if not params:
            if self.verbose > 0:
                print(f"[CURRICULUM_BASE] No parameters to set for node {node_name}")
            return True  # No parameters to set is not an error

        request = SetParameters.Request(parameters=params)
        try:
            future = client.call_async(request)
            start = time.time()

            # Use shorter polling timeout to detect hanging services
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
                    print(
                        f"[CURRICULUM_BASE] Successfully set {len(params)} parameters for {node_name}"
                    )
                return True
            else:
                if self.verbose > 0:
                    print(
                        f"[CURRICULUM_BASE] Parameter setting failed for {node_name}:"
                    )
                    if response:
                        for i, result in enumerate(response.results):
                            if not result.successful:
                                param_name = (
                                    params[i].name if i < len(params) else "unknown"
                                )
                                print(f"  - Parameter '{param_name}': {result.reason}")
                    else:
                        print("  - No response received from service")
                return False
        except Exception as e:
            if self.verbose > 0:
                print(
                    f"[CURRICULUM_BASE] Exception while setting parameters for {node_name}: {e}"
                )
            return False

    def _set_parameters(self, param_dict: Dict[str, Any]) -> bool:
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
                print(
                    f"[CURRICULUM_BASE] Successfully set parameters for: {successful_nodes}"
                )
            if failed_nodes:
                print(f"[CURRICULUM_BASE] Failed to set parameters for: {failed_nodes}")
            print(f"[CURRICULUM_BASE] Parameter setting complete: {success}")
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
            print(
                f"[CURRICULUM_BASE] Available parameter clients: {list(self.parameter_clients.keys())}"
            )
            print(
                f"[CURRICULUM_BASE] Parameter node template: {self.parameter_node_template}"
            )

        ok = self._set_parameters(stage)
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
    def get_current_performance(self) -> Optional[float]:
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

    def get_curriculum_stages(self) -> List[Dict[str, Any]]:
        """Return the normalized list of stages (stage dicts)."""
        return list(self._stages)
