from typing import Dict, Any, List
from .base import Collector, Generator


class DependencyMissingError(Exception):
    """Raised when a generator's required dependency is missing."""

    pass


class DependencyResolver:
    """Handles dependency resolution and execution ordering for generators."""

    def __init__(
        self, generators: Dict[str, Generator], collectors: Dict[str, Collector]
    ):
        self.generators = generators
        self.collectors = collectors

        self._validate_configuration()
        self.execution_order = self._resolve_dependencies()

    def _resolve_dependencies(self) -> List[str]:
        """Resolve generator dependencies using topological sorting."""
        # Build dependency graph
        graph = {name: set() for name in self.generators.keys()}

        for name, generator in self.generators.items():
            for required_key in generator.requires.keys():
                # Only add dependency if it's from another generator
                if required_key in self.generators:
                    graph[required_key].add(name)

        # Topological sort
        visited = set()
        temp_visited = set()
        result = []

        def dfs(node: str):
            if node in temp_visited:
                raise ValueError(f"Circular dependency detected involving '{node}'")
            if node in visited:
                return

            temp_visited.add(node)
            for dependent in graph[node]:
                dfs(dependent)
            temp_visited.remove(node)
            visited.add(node)
            result.append(node)

        for node in self.generators.keys():
            if node not in visited:
                dfs(node)

        return result

    def _validate_configuration(self):
        """Validate data source configuration against schemas."""
        for name, ds in self.generators.items():
            if isinstance(ds, Generator):
                # Validate required dependencies exist
                missing_deps = []
                for req_key in ds.requires.keys():
                    if (
                        req_key not in self.collectors
                        and req_key not in self.generators
                    ):
                        missing_deps.append(req_key)

                if missing_deps:
                    raise DependencyMissingError(
                        f"Generator '{name}' missing dependencies: {missing_deps}"
                    )
