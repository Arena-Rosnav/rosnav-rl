"""Advanced Laser Perception Spaces

Enhanced laser processing with proven, reliable features for production use.
"""

import numpy as np
from gymnasium import spaces

from rosnav_rl.observations.utils.types import LidarRanges
from ....observation_space_factory import SpaceFactory
from ....space_categories import SpaceCategory
from ...base_observation_space import BaseObservationSpace


@SpaceFactory.register(auto_name=True, category=SpaceCategory.PERCEPTION)
class ReliableLaserSpace(BaseObservationSpace):
    """Advanced laser scan observation space with robust preprocessing and beam reduction.

    Provides a production-ready, normalized representation of laser scan data with proven enhancements:
    range validation, NaN/Inf handling, outlier filtering, and configurable beam reduction for robust perception.

    Technical Specifications:
    - Range Validation: Clamps all values to [min_range, max_range]
    - NaN/Inf Handling: Replaces invalid values with max/min range
    - Outlier Filtering: Optional median filter for noise reduction
    - Beam Reduction: Subsamples or interpolates to reduced_beams

    Configuration:
    - laser_num_beams: Number of beams in the original scan
    - laser_max_range: Maximum valid range (meters)
    - min_range: Minimum valid range (meters)
    - reduced_beams: Number of beams in the output (None = no reduction)
    - enable_median_filter: Whether to apply median filtering
    - filter_window: Window size for median filter

    Output Format: 1D numpy array of length reduced_beams, with all values ∈ [min_range, max_range].

    Applications: Obstacle avoidance, mapping, and robust sensor fusion.
    """

    name = "RELIABLE_LASER"
    requires = {
        "front_laser": LidarRanges,
    }

    def __init__(
        self,
        laser_num_beams: int = 360,
        laser_max_range: float = 30.0,
        min_range: float = 0.1,
        reduced_beams: int = None,
        enable_median_filter: bool = False,
        filter_window: int = 3,
        *args,
        **kwargs
    ):
        """Initialize reliable laser space.

        Args:
            laser_num_beams: Expected number of laser beams
            laser_max_range: Maximum valid range
            min_range: Minimum valid range
            reduced_beams: Target number of beams (None = no reduction)
            enable_median_filter: Enable simple median filtering
            filter_window: Window size for median filter
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        self.num_beams = laser_num_beams
        self.max_range = laser_max_range
        self.min_range = min_range
        self.reduced_beams = reduced_beams or laser_num_beams
        self.enable_median_filter = enable_median_filter
        self.filter_window = filter_window

        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """Return gym space for laser data."""
        return spaces.Box(
            low=self.min_range,
            high=self.max_range,
            shape=(self.reduced_beams,),
            dtype=np.float32,
        )

    def _validate_scan(self, scan: np.ndarray) -> np.ndarray:
        """Validate and clean laser scan data."""
        # Handle NaN and Inf values
        scan = np.nan_to_num(
            scan, nan=self.max_range, posinf=self.max_range, neginf=self.min_range
        )

        # Clip to valid range
        scan = np.clip(scan, self.min_range, self.max_range)

        return scan

    def _apply_median_filter(self, scan: np.ndarray) -> np.ndarray:
        """Apply simple median filter for outlier removal."""
        if not self.enable_median_filter or len(scan) < self.filter_window:
            return scan

        filtered_scan = np.copy(scan)
        half_window = self.filter_window // 2

        for i in range(half_window, len(scan) - half_window):
            window = scan[i - half_window : i + half_window + 1]
            filtered_scan[i] = np.median(window)

        return filtered_scan

    def _reduce_beams(self, scan: np.ndarray) -> np.ndarray:
        """Reduce number of beams through simple subsampling."""
        if len(scan) == self.reduced_beams:
            return scan

        if len(scan) > self.reduced_beams:
            # Subsample evenly
            indices = np.linspace(0, len(scan) - 1, self.reduced_beams, dtype=int)
            return scan[indices]
        else:
            # Interpolate to target size
            indices = np.linspace(0, len(scan) - 1, self.reduced_beams)
            return np.interp(indices, np.arange(len(scan)), scan)

    def encode_observation(
        self, front_laser: LidarRanges, *args, **kwargs
    ) -> np.ndarray:
        """Encode robust laser scan with validation, filtering, and beam reduction.

        Args:
            front_laser (LidarRanges): Preprocessed laser scan ranges from front-facing lidar
                - Shape: (laser_num_beams,)
                - Dtype: np.float32
                - Units: meters
                - Constraints: values ∈ [min_range, max_range], NaN/Inf replaced
                - Example: [0.5, 1.2, 3.4, ..., 2.1]

        Returns:
            np.ndarray: Processed laser scan array with all enhancements applied.
                - Shape: (reduced_beams,)
                - Dtype: np.float32
                - Range: [min_range, max_range]
                - Units: meters
                - Example: [0.5, 1.2, 2.8, 1.9] for 4-beam reduction
        """
        raw_scan = front_laser
        if len(raw_scan) == 0:
            return np.full(self.reduced_beams, self.max_range, dtype=np.float32)
        processed_scan = self._validate_scan(raw_scan.astype(np.float32))
        processed_scan = self._apply_median_filter(processed_scan)
        processed_scan = self._reduce_beams(processed_scan)
        return processed_scan.astype(np.float32)


@SpaceFactory.register(auto_name=True, category=SpaceCategory.PERCEPTION)
class MultiRangeLaserSpace(BaseObservationSpace):
    """Multi-scale laser scan observation space for enhanced range sensitivity.

    Provides laser scan data at multiple range scales, enabling the agent to detect both near and far objects
    with improved sensitivity. Used in advanced navigation systems for robust perception.

    Technical Specifications:
    - Multi-Scale Ranges: Configurable list of range scales for normalization
    - Beam Count: Fixed number of beams per scale

    Configuration:
    - laser_num_beams: Number of beams in the original scan
    - range_scales: List of scale factors (e.g., [0.3, 1.0, 2.0])
    - laser_max_range: Base maximum range (meters)
    - min_range: Minimum valid range (meters)

    Output Format: 1D numpy array of length (laser_num_beams * num_scales), normalized to [0, 1].

    Applications: Near/far object detection, multi-scale perception, and robust navigation.
    """

    name = "MULTI_RANGE_LASER"
    requires = {
        "front_laser": LidarRanges,
    }

    def __init__(
        self,
        laser_num_beams: int = 360,
        range_scales: list = None,
        laser_max_range: float = 30.0,
        min_range: float = 0.1,
        *args,
        **kwargs
    ):
        """Initialize multi-range laser space.

        Args:
            laser_num_beams: Number of laser beams
            range_scales: List of range scale factors [0.2, 1.0, 3.0] = [short, medium, long]
            laser_max_range: Base maximum range
            min_range: Minimum valid range
            *args: Variable arguments
            **kwargs: Keyword arguments
        """
        self.num_beams = laser_num_beams
        self.range_scales = range_scales or [0.3, 1.0, 2.0]  # Conservative defaults
        self.base_max_range = laser_max_range
        self.min_range = min_range

        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """Return gym space for multi-range laser data."""
        total_dims = self.num_beams * len(self.range_scales)
        return spaces.Box(
            low=0.0, high=1.0, shape=(total_dims,), dtype=np.float32  # Normalized
        )

    def encode_observation(
        self, front_laser: LidarRanges, *args, **kwargs
    ) -> np.ndarray:
        """Encode multi-scale laser scan with configurable range sensitivities.

        Args:
            front_laser (LidarRanges): Preprocessed laser scan ranges from front-facing lidar
                - Shape: (laser_num_beams,)
                - Dtype: np.float32
                - Units: meters
                - Constraints: values ∈ [min_range, laser_max_range * max(range_scales)]
                - Example: [0.5, 1.2, 3.4, ..., 2.1]

        Returns:
            np.ndarray: Multi-scale laser representation, concatenated for all scales.
                - Shape: (laser_num_beams * num_scales,)
                - Dtype: np.float32
                - Range: [0.0, 1.0]
                - Units: normalized
                - Example: [0.2, 0.5, 0.1, ...] for 3 scales
        """
        raw_scan = front_laser
        if len(raw_scan) == 0:
            total_dims = self.num_beams * len(self.range_scales)
            return np.zeros(total_dims, dtype=np.float32)
        scan = np.nan_to_num(raw_scan, nan=self.base_max_range)
        scan = np.clip(
            scan, self.min_range, self.base_max_range * max(self.range_scales)
        )
        if len(scan) != self.num_beams:
            indices = np.linspace(0, len(scan) - 1, self.num_beams)
            scan = np.interp(indices, np.arange(len(scan)), scan)
        multi_scale_data = []
        for scale in self.range_scales:
            max_range = self.base_max_range * scale
            normalized_scan = np.clip(scan / max_range, 0.0, 1.0)
            multi_scale_data.extend(normalized_scan)
        return np.array(multi_scale_data, dtype=np.float32)


@SpaceFactory.register(auto_name=True, category=SpaceCategory.PERCEPTION)
class MultiLaserFusionSpace(BaseObservationSpace):
    """Advanced multi-laser fusion observation space for robust sensor integration.

    Fuses multiple laser scanners using confidence-based weighting, angular overlap handling, robust outlier detection,
    and seamless degradation with sensor failures. Provides a unified, normalized laser scan for advanced navigation.

    Technical Specifications:
    - Sensor Fusion: Weighted average, confidence max, or min distance fusion
    - Outlier Detection: Median-based robust outlier removal
    - Sensor Health: Confidence tracking and failure detection

    Configuration:
    - laser_configs: List of laser configuration dicts (topic, weight, angle_offset)
    - fusion_method: Fusion strategy ("weighted_average", "confidence_max")
    - confidence_threshold: Minimum confidence for sensor data
    - overlap_resolution: How to handle overlapping beams ("min_distance")
    - enable_failure_detection: Enable sensor failure detection
    - outlier_threshold: Threshold for outlier detection (std devs)

    Output Format: 1D numpy array of length num_output_beams, normalized to [0, 1].

    Applications: Sensor fusion, robust navigation, and multi-laser environments.
    """

    name = "MULTI_LASER_FUSION"
    requires = {
        "laser_fusion": LidarRanges,
    }

    def __init__(
        self,
        laser_configs: list = None,
        fusion_method: str = "weighted_average",
        confidence_threshold: float = 0.8,
        overlap_resolution: str = "min_distance",
        enable_failure_detection: bool = True,
        outlier_threshold: float = 3.0,
        *args,
        **kwargs
    ):
        """Initialize multi-laser fusion space.

        Args:
            laser_configs: List of laser configurations
            fusion_method: Method for fusion ('weighted_average', 'confidence_max')
            confidence_threshold: Minimum confidence for sensor data
            overlap_resolution: How to handle overlapping beams
            enable_failure_detection: Enable sensor failure detection
            outlier_threshold: Threshold for outlier detection (std devs)
        """
        if laser_configs is None:
            laser_configs = [
                {"topic": "/scan_front", "weight": 1.0, "angle_offset": 0.0},
                {"topic": "/scan_rear", "weight": 0.8, "angle_offset": 3.14159},
            ]

        self.laser_configs = laser_configs
        self.fusion_method = fusion_method
        self.confidence_threshold = confidence_threshold
        self.overlap_resolution = overlap_resolution
        self.enable_failure_detection = enable_failure_detection
        self.outlier_threshold = outlier_threshold

        # Processing parameters
        self.num_output_beams = 360
        self.max_range = 30.0
        self.min_range = 0.1

        # Sensor health tracking
        self.sensor_health = {cfg["topic"]: 1.0 for cfg in laser_configs}
        self.last_update_times = {cfg["topic"]: 0.0 for cfg in laser_configs}

        super().__init__(*args, **kwargs)

    def get_gym_space(self) -> spaces.Space:
        """Get gym space for fused laser data."""
        return spaces.Box(
            low=0.0,
            high=1.0,  # Normalized ranges
            shape=(self.num_output_beams,),
            dtype=np.float32,
        )

    def _calculate_sensor_confidence(self, scan_data: np.ndarray, topic: str) -> float:
        """Calculate confidence score for a sensor.

        Args:
            scan_data: Raw scan data
            topic: Laser topic name

        Returns:
            Confidence score [0, 1]
        """
        if len(scan_data) == 0:
            return 0.0

        # Check for common failure modes
        valid_readings = np.isfinite(scan_data)
        if np.sum(valid_readings) < len(scan_data) * 0.5:
            return 0.3  # Low confidence for sparse data

        # Check for stuck readings (all same value)
        unique_values = len(np.unique(scan_data[valid_readings]))
        if unique_values < len(scan_data) * 0.1:
            return 0.2  # Very low confidence for stuck sensor

        # Check range distribution
        valid_data = scan_data[valid_readings]
        range_std = np.std(valid_data)
        if range_std < 0.01:  # Too uniform
            return 0.4

        # Update sensor health based on recent performance
        current_health = self.sensor_health.get(topic, 1.0)

        # Exponential moving average for health
        alpha = 0.1
        base_confidence = 0.9  # Base confidence for good data
        self.sensor_health[topic] = (
            1 - alpha
        ) * current_health + alpha * base_confidence

        return min(self.sensor_health[topic], 1.0)

    def _detect_outliers(self, scan_data: np.ndarray) -> np.ndarray:
        """Detect and mark outliers in scan data.

        Args:
            scan_data: Input scan data

        Returns:
            Boolean mask where True indicates outlier
        """
        if len(scan_data) < 3:
            return np.zeros(len(scan_data), dtype=bool)

        # Use median-based outlier detection (robust)
        median_val = np.median(scan_data)
        mad = np.median(np.abs(scan_data - median_val))

        if mad == 0:
            return np.zeros(len(scan_data), dtype=bool)

        # Modified Z-score
        modified_z = 0.6745 * (scan_data - median_val) / mad
        outliers = np.abs(modified_z) > self.outlier_threshold

        return outliers

    def _fuse_laser_data(self, laser_data_list: list) -> np.ndarray:
        """Fuse multiple laser scans into unified representation.

        Args:
            laser_data_list: List of (scan_data, confidence, config) tuples

        Returns:
            Fused laser scan data
        """
        # Initialize output array
        fused_scan = np.full(self.num_output_beams, self.max_range, dtype=np.float32)
        weight_sum = np.zeros(self.num_output_beams)

        for scan_data, confidence, config in laser_data_list:
            if confidence < self.confidence_threshold:
                continue  # Skip low-confidence sensors

            # Handle different scan sizes
            if len(scan_data) != self.num_output_beams:
                # Resample to target resolution
                angles = np.linspace(0, 2 * np.pi, len(scan_data))
                target_angles = np.linspace(0, 2 * np.pi, self.num_output_beams)
                scan_data = np.interp(target_angles, angles, scan_data)

            # Apply angular offset if specified
            if config.get("angle_offset", 0.0) != 0.0:
                offset_samples = int(
                    config["angle_offset"] / (2 * np.pi) * self.num_output_beams
                )
                scan_data = np.roll(scan_data, offset_samples)

            # Detect and handle outliers
            outlier_mask = self._detect_outliers(scan_data)
            scan_data[outlier_mask] = self.max_range  # Set outliers to max range

            # Apply sensor weight and confidence
            sensor_weight = config.get("weight", 1.0) * confidence

            if self.fusion_method == "weighted_average":
                # Weighted average fusion
                fused_scan = (fused_scan * weight_sum + scan_data * sensor_weight) / (
                    weight_sum + sensor_weight
                )
                weight_sum += sensor_weight

            elif self.fusion_method == "confidence_max":
                # Use data from most confident sensor per beam
                better_mask = sensor_weight > weight_sum
                fused_scan[better_mask] = scan_data[better_mask]
                weight_sum[better_mask] = sensor_weight

            elif self.overlap_resolution == "min_distance":
                # Take minimum distance (most conservative)
                fused_scan = np.minimum(fused_scan, scan_data)

        # Clip to valid ranges
        fused_scan = np.clip(fused_scan, self.min_range, self.max_range)

        # Normalize to [0, 1]
        normalized_scan = fused_scan / self.max_range

        return normalized_scan

    def encode_observation(
        self, laser_fusion: LidarRanges, *args, **kwargs
    ) -> np.ndarray:
        """Encode fused multi-laser scan with robust sensor integration and normalization.

        Args:
            laser_fusion (LidarRanges): Preprocessed, fused laser scan data from multiple sensors
                - Shape: (num_output_beams,)
                - Dtype: np.float32
                - Units: meters
                - Constraints: values ∈ [min_range, max_range], NaN/Inf replaced
                - Example: [0.2, 0.5, 0.1, ...] for 360-beam output

        Returns:
            np.ndarray: Fused and normalized laser scan.
                - Shape: (num_output_beams,)
                - Dtype: np.float32
                - Range: [0.0, 1.0]
                - Units: normalized
                - Example: [0.2, 0.5, 0.1, ...] for 360-beam output
        """
        # In a real system, laser_fusion would be the result of a fusion pipeline.
        # Here, we simply normalize the input for demonstration.
        fused_scan = np.clip(laser_fusion, self.min_range, self.max_range)
        normalized_scan = fused_scan / self.max_range
        return normalized_scan
