from typing import Dict, Any, List
from ..data_sources.base import Collector, Generator


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

    def get_dependency_info(self) -> Dict[str, Any]:
        """Get debugging information about dependencies."""
        info = {
            "execution_order": self.execution_order,
            "dependencies": {},
            "dependency_graph": {},
        }

        for name, generator in self.generators.items():
            deps = [req for req in generator.requires.keys() if req in self.generators]
            info["dependencies"][name] = deps

        # Build dependency graph for visualization
        for name in self.generators.keys():
            dependents = []
            for other_name, other_gen in self.generators.items():
                if name in other_gen.requires.keys():
                    dependents.append(other_name)
            info["dependency_graph"][name] = dependents

        return info

    def _resolve_dependencies(self) -> List[str]:
        """Resolve generator dependencies using topological sorting."""
        # Build dependency graph: generator_name -> set of generators it depends on
        dependencies = {name: set() for name in self.generators.keys()}

        for name, generator in self.generators.items():
            for required_key in generator.requires.keys():
                # Only add dependency if it's from another generator
                if required_key in self.generators:
                    dependencies[name].add(required_key)

        # Topological sort using Kahn's algorithm (more straightforward for this case)
        # Calculate in-degrees (number of dependencies)
        in_degree = {name: len(deps) for name, deps in dependencies.items()}

        # Start with generators that have no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # Process a generator with no remaining dependencies
            current = queue.pop(0)
            result.append(current)

            # Remove this generator as a dependency from others
            for name, deps in dependencies.items():
                if current in deps:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)

        # Check for circular dependencies
        if len(result) != len(self.generators):
            remaining = set(self.generators.keys()) - set(result)
            raise ValueError(
                f"Circular dependency detected among generators: {remaining}"
            )

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
