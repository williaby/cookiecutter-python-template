"""
Configuration system for task detection algorithm

Provides configurable parameters for tuning detection accuracy,
performance characteristics, and fallback behavior.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class DetectionMode(Enum):
    """Detection operation modes"""

    CONSERVATIVE = "conservative"  # Favor over-inclusion for safety
    BALANCED = "balanced"  # Balance accuracy and performance
    AGGRESSIVE = "aggressive"  # Favor under-inclusion for performance


class FallbackStrategy(Enum):
    """Fallback strategies for ambiguous detection"""

    SAFE_DEFAULT = "safe_default"  # Load core + analysis
    EXPAND_ALL = "expand_all"  # Load all tier 2 categories
    CONTEXT_BASED = "context_based"  # Use context for decisions
    FULL_LOAD = "full_load"  # Load everything except tier 3


@dataclass
class PerformanceConfig:
    """Performance-related configuration"""

    max_detection_time_ms: float = 50.0
    max_memory_usage_mb: int = 10
    cache_size: int = 1000
    cache_ttl_hours: int = 1
    enable_parallel_signals: bool = True
    signal_timeout_ms: float = 10.0


@dataclass
class ThresholdConfig:
    """Detection threshold configuration"""

    tier2_base_threshold: float = 0.25  # Conservative threshold for over-inclusion safety
    tier3_base_threshold: float = 0.55  # Conservative threshold for tier 3
    high_confidence_threshold: float = 0.9
    medium_confidence_threshold: float = 0.6
    ambiguous_difference_threshold: float = 0.15


@dataclass
class BiasConfig:
    """Conservative bias configuration"""

    new_user_threshold_multiplier: float = 0.6
    complex_query_threshold_multiplier: float = 0.7
    multi_domain_threshold_multiplier: float = 0.6
    error_context_boost: float = 0.2
    enable_conservative_expansion: bool = True


@dataclass
class SignalWeightConfig:
    """Signal weighting configuration"""

    keyword_direct: float = 1.0
    keyword_contextual: float = 0.7
    keyword_action: float = 0.5
    context_files: float = 0.6
    context_errors: float = 0.8
    context_performance: float = 0.7
    environment_git: float = 0.7
    environment_structure: float = 0.5
    session_recent: float = 0.6
    session_pattern: float = 0.8


@dataclass
class CategoryModifierConfig:
    """Category-specific modifier configuration"""

    git_keyword_direct: float = 1.2
    debug_context_errors: float = 1.3
    security_context_files: float = 1.1
    test_environment_structure: float = 1.2
    analysis_session_pattern: float = 1.1


@dataclass
class TierDefinitionConfig:
    """Function tier definitions"""

    tier1_categories: list[str] = field(default_factory=lambda: ["core", "git"])
    tier2_categories: list[str] = field(default_factory=lambda: ["analysis", "quality", "debug", "test", "security"])
    tier3_categories: list[str] = field(default_factory=lambda: ["external", "infrastructure"])

    tier1_token_cost: int = 12300
    tier2_token_cost: int = 19540
    tier3_token_cost: int = 5800


@dataclass
class CalibrationConfig:
    """Confidence calibration configuration"""

    enable_calibration: bool = True
    calibration_curves: dict[str, dict[float, float]] = field(
        default_factory=lambda: {
            "git": {0.3: 0.8, 0.5: 0.9, 0.7: 0.95, 1.0: 1.0},
            "debug": {0.3: 0.7, 0.5: 0.8, 0.7: 0.9, 1.0: 0.95},
            "test": {0.3: 0.6, 0.5: 0.75, 0.7: 0.85, 1.0: 0.9},
            "security": {0.3: 0.7, 0.5: 0.8, 0.7: 0.9, 1.0: 0.95},
            "analysis": {0.3: 0.5, 0.5: 0.65, 0.7: 0.8, 1.0: 0.9},
            "quality": {0.3: 0.6, 0.5: 0.7, 0.7: 0.85, 1.0: 0.9},
        },
    )

    complexity_high_threshold: float = 0.8
    complexity_low_threshold: float = 0.3
    complexity_high_modifier: float = 0.8
    complexity_low_modifier: float = 1.1


@dataclass
class LearningConfig:
    """Learning system configuration"""

    enable_learning: bool = True
    min_samples_for_adaptation: int = 100
    weight_adaptation_rate: float = 0.1
    max_weight_change: float = 0.2
    accuracy_target_f1: float = 0.8
    pattern_history_limit: int = 1000
    user_pattern_limit: int = 1000


@dataclass
class FallbackConfig:
    """Fallback behavior configuration"""

    default_strategy: FallbackStrategy = FallbackStrategy.CONTEXT_BASED
    enable_context_adjustments: bool = True
    security_project_enables_security: bool = True
    test_dirs_enable_testing: bool = True
    code_files_enable_quality: bool = True
    expansion_category_limit: int = 3
    expansion_min_score: float = 0.2


@dataclass
class KeywordConfig:
    """Keyword detection configuration"""

    enable_fuzzy_matching: bool = False
    fuzzy_threshold: float = 0.8
    case_sensitive: bool = False
    word_boundary_required: bool = True

    # Custom keyword patterns can be added here
    custom_patterns: dict[str, dict[str, list[str]]] = field(default_factory=dict)


@dataclass
class TaskDetectionConfig:
    """Main configuration container for task detection system"""

    # Operation mode
    mode: DetectionMode = DetectionMode.CONSERVATIVE

    # Component configurations
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    bias: BiasConfig = field(default_factory=BiasConfig)
    signal_weights: SignalWeightConfig = field(default_factory=SignalWeightConfig)
    category_modifiers: CategoryModifierConfig = field(default_factory=CategoryModifierConfig)
    tier_definitions: TierDefinitionConfig = field(default_factory=TierDefinitionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    keywords: KeywordConfig = field(default_factory=KeywordConfig)

    # Metadata
    version: str = "1.0.0"
    description: str = "Task detection configuration for dynamic function loading"

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary"""

        def convert_enum(obj):
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, dict):
                return {k: convert_enum(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_enum(item) for item in obj]
            return obj

        # Convert dataclass to dict
        result = {}
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, "__dict__"):
                # Nested dataclass
                result[field_name] = {k: convert_enum(v) for k, v in field_value.__dict__.items()}
            else:
                result[field_name] = convert_enum(field_value)

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskDetectionConfig":
        """Create configuration from dictionary"""
        config = cls()

        # Handle mode enum
        if "mode" in data:
            config.mode = DetectionMode(data["mode"])

        # Handle nested configurations
        nested_configs = {
            "performance": PerformanceConfig,
            "thresholds": ThresholdConfig,
            "bias": BiasConfig,
            "signal_weights": SignalWeightConfig,
            "category_modifiers": CategoryModifierConfig,
            "tier_definitions": TierDefinitionConfig,
            "calibration": CalibrationConfig,
            "learning": LearningConfig,
            "fallback": FallbackConfig,
            "keywords": KeywordConfig,
        }

        for config_name, config_class in nested_configs.items():
            if config_name in data:
                config_data = data[config_name]

                # Handle enum fields in nested configs
                if config_name == "fallback" and "default_strategy" in config_data:
                    config_data["default_strategy"] = FallbackStrategy(config_data["default_strategy"])

                # Create nested config object
                setattr(config, config_name, config_class(**config_data))

        # Handle simple fields
        simple_fields = ["version", "description"]
        for field_name in simple_fields:
            if field_name in data:
                setattr(config, field_name, data[field_name])

        return config

    def save_to_file(self, file_path: str | Path) -> None:
        """Save configuration to file (JSON or YAML)"""
        file_path = Path(file_path)
        data = self.to_dict()

        if file_path.suffix.lower() == ".yaml" or file_path.suffix.lower() == ".yml":
            with open(file_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
        else:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: str | Path) -> "TaskDetectionConfig":
        """Load configuration from file (JSON or YAML)"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        if file_path.suffix.lower() == ".yaml" or file_path.suffix.lower() == ".yml":
            with open(file_path) as f:
                data = yaml.safe_load(f)
        else:
            with open(file_path) as f:
                data = json.load(f)

        return cls.from_dict(data)

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues"""
        issues = []

        # Validate performance constraints
        if self.performance.max_detection_time_ms <= 0:
            issues.append("max_detection_time_ms must be positive")

        if self.performance.max_memory_usage_mb <= 0:
            issues.append("max_memory_usage_mb must be positive")

        # Validate thresholds
        if not 0.0 <= self.thresholds.tier2_base_threshold <= 1.0:
            issues.append("tier2_base_threshold must be between 0.0 and 1.0")

        if not 0.0 <= self.thresholds.tier3_base_threshold <= 1.0:
            issues.append("tier3_base_threshold must be between 0.0 and 1.0")

        if self.thresholds.tier2_base_threshold >= self.thresholds.tier3_base_threshold:
            issues.append("tier2_base_threshold must be less than tier3_base_threshold")

        # Validate signal weights
        weight_fields = [
            "keyword_direct",
            "keyword_contextual",
            "keyword_action",
            "context_files",
            "context_errors",
            "context_performance",
            "environment_git",
            "environment_structure",
            "session_recent",
            "session_pattern",
        ]

        for field in weight_fields:
            value = getattr(self.signal_weights, field, 0)
            if not 0.0 <= value <= 2.0:
                issues.append(f"signal_weights.{field} should be between 0.0 and 2.0")

        # Validate tier definitions
        all_categories = (
            self.tier_definitions.tier1_categories
            + self.tier_definitions.tier2_categories
            + self.tier_definitions.tier3_categories
        )

        if len(set(all_categories)) != len(all_categories):
            issues.append("Duplicate categories found across tiers")

        # Validate calibration curves
        for category, curve in self.calibration.calibration_curves.items():
            sorted_points = sorted(curve.items())
            for i, (threshold, adjustment) in enumerate(sorted_points):
                if not 0.0 <= threshold <= 1.0:
                    issues.append(f"Calibration threshold for {category} out of range: {threshold}")

                if adjustment <= 0:
                    issues.append(f"Calibration adjustment for {category} must be positive: {adjustment}")

                if i > 0:
                    prev_threshold, prev_adjustment = sorted_points[i - 1]
                    if threshold <= prev_threshold:
                        issues.append(f"Calibration thresholds for {category} not strictly increasing")

        return issues

    def apply_mode_preset(self, mode: DetectionMode) -> None:
        """Apply preset configurations for different modes"""
        self.mode = mode

        if mode == DetectionMode.CONSERVATIVE:
            # Conservative: favor over-inclusion
            self.thresholds.tier2_base_threshold = 0.25
            self.thresholds.tier3_base_threshold = 0.55
            self.bias.new_user_threshold_multiplier = 0.6
            self.bias.complex_query_threshold_multiplier = 0.7
            self.bias.enable_conservative_expansion = True
            self.fallback.default_strategy = FallbackStrategy.CONTEXT_BASED

        elif mode == DetectionMode.BALANCED:
            # Balanced: default settings
            self.thresholds.tier2_base_threshold = 0.3
            self.thresholds.tier3_base_threshold = 0.6
            self.bias.new_user_threshold_multiplier = 0.7
            self.bias.complex_query_threshold_multiplier = 0.8
            self.bias.enable_conservative_expansion = True
            self.fallback.default_strategy = FallbackStrategy.CONTEXT_BASED

        elif mode == DetectionMode.AGGRESSIVE:
            # Aggressive: favor under-inclusion for performance
            self.thresholds.tier2_base_threshold = 0.4
            self.thresholds.tier3_base_threshold = 0.7
            self.bias.new_user_threshold_multiplier = 0.8
            self.bias.complex_query_threshold_multiplier = 0.9
            self.bias.enable_conservative_expansion = False
            self.fallback.default_strategy = FallbackStrategy.SAFE_DEFAULT

    def get_environment_preset(self, environment: str) -> "TaskDetectionConfig":
        """Get configuration preset for specific environments"""
        config = TaskDetectionConfig()

        if environment == "development":
            # Development: prioritize functionality over performance
            config.apply_mode_preset(DetectionMode.CONSERVATIVE)
            config.performance.max_detection_time_ms = 100.0  # More lenient
            config.learning.enable_learning = True

        elif environment == "production":
            # Production: balance functionality and performance
            config.apply_mode_preset(DetectionMode.BALANCED)
            config.performance.max_detection_time_ms = 50.0
            config.learning.enable_learning = True

        elif environment == "performance_critical":
            # Performance critical: prioritize speed
            config.apply_mode_preset(DetectionMode.AGGRESSIVE)
            config.performance.max_detection_time_ms = 30.0
            config.performance.enable_parallel_signals = True
            config.learning.enable_learning = False  # Disable for consistency

        elif environment == "new_user":
            # New user: maximum safety
            config.apply_mode_preset(DetectionMode.CONSERVATIVE)
            config.bias.new_user_threshold_multiplier = 0.5
            config.fallback.default_strategy = FallbackStrategy.EXPAND_ALL

        return config


class ConfigManager:
    """Manages configuration loading and validation"""

    def __init__(self, config_dir: str | Path | None = None) -> None:
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self.config_dir.mkdir(exist_ok=True)

        self._config_cache: dict[str, TaskDetectionConfig] = {}

    def get_config(self, config_name: str = "default") -> TaskDetectionConfig:
        """Get configuration by name"""
        if config_name in self._config_cache:
            return self._config_cache[config_name]

        config_file = self.config_dir / f"{config_name}.yaml"

        if config_file.exists():
            config = TaskDetectionConfig.load_from_file(config_file)
        else:
            # Create default configuration
            config = TaskDetectionConfig()
            if config_name != "default":
                # Try to apply environment preset
                try:
                    config = config.get_environment_preset(config_name)
                except Exception:  # nosec B110
                    pass  # Use default if preset fails

            # Save default configuration
            config.save_to_file(config_file)

        # Validate configuration
        issues = config.validate()
        if issues:
            raise ValueError(f"Configuration validation failed: {', '.join(issues)}")

        self._config_cache[config_name] = config
        return config

    def save_config(self, config: TaskDetectionConfig, config_name: str = "default") -> None:
        """Save configuration with name"""
        config_file = self.config_dir / f"{config_name}.yaml"
        config.save_to_file(config_file)
        self._config_cache[config_name] = config

    def list_configs(self) -> list[str]:
        """List available configuration names"""
        config_files = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.json"))
        return [f.stem for f in config_files]

    def create_preset_configs(self) -> None:
        """Create standard preset configurations"""
        presets = {
            "development": "development",
            "production": "production",
            "performance_critical": "performance_critical",
            "new_user": "new_user",
        }

        base_config = TaskDetectionConfig()

        for preset_name, environment in presets.items():
            preset_config = base_config.get_environment_preset(environment)
            self.save_config(preset_config, preset_name)


# Default configuration instance
default_config = TaskDetectionConfig()


# Example usage and configuration generation
if __name__ == "__main__":
    # Create config manager and generate presets
    manager = ConfigManager()
    manager.create_preset_configs()

    # Create a custom configuration
    custom_config = TaskDetectionConfig()
    custom_config.apply_mode_preset(DetectionMode.CONSERVATIVE)
    custom_config.performance.max_detection_time_ms = 75.0
    custom_config.learning.enable_learning = True

    # Save custom configuration
    manager.save_config(custom_config, "custom")

    for _config_name in manager.list_configs():
        pass

    # Load and validate a configuration
    config = manager.get_config("production")

    # Validate configuration
    issues = config.validate()
    if issues:
        pass
    else:
        pass

    # Show configuration as dict
    config_dict = config.to_dict()
