"""Multi-Laser Fusion Production Example

Demonstrates best practices for using MultiLaserFusionSpace in production environments.
"""

import numpy as np
from typing import Dict, Any

from rosnav_rl.spaces.observation_space.spaces.perception.laser import (
    MultiLaserFusionSpace,
)


class ProductionMultiLaserSetup:
    """Production-ready multi-laser configuration examples."""

    @staticmethod
    def create_dual_laser_config() -> Dict[str, Any]:
        """Standard dual-laser setup for robots with front/rear scanners.

        Returns:
            Configuration dictionary for MultiLaserFusionSpace
        """
        return {
            "laser_configs": [
                {"topic": "/scan_front", "weight": 1.0, "angle_offset": 0.0},
                {
                    "topic": "/scan_rear",
                    "weight": 0.9,  # Slightly lower weight for rear
                    "angle_offset": 3.14159,  # 180 degrees
                },
            ],
            "fusion_method": "weighted_average",
            "confidence_threshold": 0.7,
            "overlap_resolution": "min_distance",
            "enable_failure_detection": True,
            "outlier_threshold": 2.5,  # Conservative for safety
        }

    @staticmethod
    def create_quad_laser_config() -> Dict[str, Any]:
        """Advanced quad-laser setup for high-reliability systems.

        Returns:
            Configuration dictionary for quad-laser fusion
        """
        return {
            "laser_configs": [
                {"topic": "/scan_front", "weight": 1.0, "angle_offset": 0.0},
                {
                    "topic": "/scan_right",
                    "weight": 0.95,
                    "angle_offset": 1.5708,
                },  # 90 degrees
                {
                    "topic": "/scan_rear",
                    "weight": 0.9,
                    "angle_offset": 3.14159,
                },  # 180 degrees
                {
                    "topic": "/scan_left",
                    "weight": 0.95,
                    "angle_offset": 4.71239,
                },  # 270 degrees
            ],
            "fusion_method": "confidence_max",  # Take best sensor per direction
            "confidence_threshold": 0.8,
            "overlap_resolution": "min_distance",
            "enable_failure_detection": True,
            "outlier_threshold": 2.0,
        }

    @staticmethod
    def create_warehouse_config() -> Dict[str, Any]:
        """Configuration optimized for warehouse environments.

        Returns:
            Warehouse-specific configuration
        """
        return {
            "laser_configs": [
                {
                    "topic": "/scan_safety",
                    "weight": 1.2,
                    "angle_offset": 0.0,
                },  # Higher weight for safety
                {"topic": "/scan_nav", "weight": 1.0, "angle_offset": 0.0},
            ],
            "fusion_method": "min_distance",  # Most conservative for safety
            "confidence_threshold": 0.9,  # High threshold for warehouse safety
            "overlap_resolution": "min_distance",
            "enable_failure_detection": True,
            "outlier_threshold": 1.5,  # Very conservative
        }


class MultiLaserProductionValidator:
    """Validation utilities for production multi-laser systems."""

    @staticmethod
    def validate_fusion_output(
        fused_scan: np.ndarray, expected_beams: int = 360
    ) -> Dict[str, Any]:
        """Validate fused laser output for production use.

        Args:
            fused_scan: Fused laser scan data
            expected_beams: Expected number of beams

        Returns:
            Validation report
        """
        report = {"valid": True, "issues": [], "statistics": {}}

        # Check shape
        if len(fused_scan) != expected_beams:
            report["valid"] = False
            report["issues"].append(
                f"Wrong beam count: {len(fused_scan)} != {expected_beams}"
            )

        # Check data range
        if np.any(fused_scan < 0) or np.any(fused_scan > 1):
            report["valid"] = False
            report["issues"].append("Data out of normalized range [0,1]")

        # Check for NaN/Inf
        if np.any(~np.isfinite(fused_scan)):
            report["valid"] = False
            report["issues"].append("NaN or Inf values detected")

        # Statistical checks
        valid_data = fused_scan[np.isfinite(fused_scan)]
        if len(valid_data) > 0:
            report["statistics"] = {
                "mean": float(np.mean(valid_data)),
                "std": float(np.std(valid_data)),
                "min": float(np.min(valid_data)),
                "max": float(np.max(valid_data)),
                "valid_ratio": len(valid_data) / len(fused_scan),
            }

            # Check for reasonable statistics
            if report["statistics"]["valid_ratio"] < 0.8:
                report["issues"].append("Low valid data ratio")

            if report["statistics"]["std"] < 0.01:
                report["issues"].append(
                    "Suspiciously low variance - possible stuck sensor"
                )

        return report

    @staticmethod
    def benchmark_fusion_performance(
        space: MultiLaserFusionSpace, num_samples: int = 1000
    ) -> Dict[str, float]:
        """Benchmark fusion performance.

        Args:
            space: MultiLaserFusionSpace instance
            num_samples: Number of samples for benchmarking

        Returns:
            Performance metrics
        """
        import time

        # Create sample observation
        sample_obs = {"laser": np.random.uniform(0.1, 30.0, 360).astype(np.float32)}

        # Warm up
        for _ in range(10):
            space.encode_observation(sample_obs)

        # Benchmark
        start_time = time.time()
        for _ in range(num_samples):
            space.encode_observation(sample_obs)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time = total_time / num_samples

        return {
            "total_time_ms": total_time * 1000,
            "avg_time_ms": avg_time * 1000,
            "fps": 1.0 / avg_time if avg_time > 0 else float("inf"),
        }


def demonstrate_production_usage():
    """Demonstrate production usage of multi-laser fusion."""

    print("=== Multi-Laser Fusion Production Demo ===\n")

    # 1. Create production configuration
    config = ProductionMultiLaserSetup.create_dual_laser_config()
    print("1. Created dual-laser configuration:")
    print(f"   Fusion method: {config['fusion_method']}")
    print(f"   Confidence threshold: {config['confidence_threshold']}")
    print(f"   Number of sensors: {len(config['laser_configs'])}\n")

    # 2. Initialize fusion space
    fusion_space = MultiLaserFusionSpace(**config)
    print("2. Initialized MultiLaserFusionSpace")
    print(f"   Gym space: {fusion_space.get_gym_space()}\n")

    # 3. Test with sample data
    sample_observation = {"laser": np.random.uniform(0.5, 25.0, 360).astype(np.float32)}

    fused_result = fusion_space.encode_observation(sample_observation)
    print("3. Processed sample observation:")
    print(f"   Input shape: {sample_observation['laser'].shape}")
    print(f"   Output shape: {fused_result.shape}")
    print(f"   Output range: [{fused_result.min():.3f}, {fused_result.max():.3f}]\n")

    # 4. Validate output
    validation = MultiLaserProductionValidator.validate_fusion_output(fused_result)
    print("4. Validation results:")
    print(f"   Valid: {validation['valid']}")
    if validation["issues"]:
        print(f"   Issues: {validation['issues']}")
    print(f"   Statistics: {validation['statistics']}\n")

    # 5. Performance benchmark
    performance = MultiLaserProductionValidator.benchmark_fusion_performance(
        fusion_space
    )
    print("5. Performance benchmark:")
    print(f"   Average processing time: {performance['avg_time_ms']:.3f} ms")
    print(f"   Throughput: {performance['fps']:.1f} FPS\n")

    print("=== Demo Complete ===")


if __name__ == "__main__":
    demonstrate_production_usage()
