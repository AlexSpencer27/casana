from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


@dataclass
class EarlyStoppingConfig:
    patience: int = 10
    min_delta: float = 0.0001


@dataclass
class TrainingConfig:
    num_epochs: int = 2000
    batch_size: int = 32
    learning_rate: float = 0.0001
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()


@dataclass
class SignalConfig:
    length: int = 2048
    sampling_rate: int = 1024
    add_complex_signal: bool = True


@dataclass
class VisualizationConfig:
    num_predictions: int = 32
    plot_frequency: Optional[int] = None  # If None, only plot final loss figure


@dataclass
class GradientRefinementConfig:
    """Configuration for gradient-based peak refinement.
    
    Attributes:
        enabled: Whether to use gradient refinement
        base_step_size: Base learning rate for gradient updates (default: 0.002)
        max_iterations: Maximum number of refinement iterations (default: 7)
    """
    enabled: bool = False
    base_step_size: float = 0.002
    max_iterations: int = 7


@dataclass
class ModelConfig:
    name: str = "dual_pathway_fft"
    gradient_refinement: GradientRefinementConfig = GradientRefinementConfig()


@dataclass
class LossConfig:
    name: str = "simple_mse"
    # Common weights
    position_weight: float = 1.0
    magnitude_weight: float = 0.5
    # Gradient-aware specific weights
    gradient_weight: float = 0.3
    second_derivative_weight: float = 0.2


@dataclass
class Config:
    training: TrainingConfig = TrainingConfig()
    signal: SignalConfig = SignalConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create a Config instance from a dictionary."""
        training_dict = config_dict.get('training', {})
        early_stopping_dict = training_dict.get('early_stopping', {})
        training_dict['early_stopping'] = EarlyStoppingConfig(**early_stopping_dict)
        
        model_dict = config_dict.get('model', {})
        gradient_refinement_dict = model_dict.get('gradient_refinement', {})
        model_dict['gradient_refinement'] = GradientRefinementConfig(**gradient_refinement_dict)
        
        return cls(
            training=TrainingConfig(**training_dict),
            signal=SignalConfig(**config_dict.get('signal', {})),
            visualization=VisualizationConfig(**config_dict.get('visualization', {})),
            model=ModelConfig(**model_dict),
            loss=LossConfig(**config_dict.get('loss', {}))
        )


def load_config() -> Config:
    """Load configuration from YAML file or return default config if file not found."""
    config_path = Path(__file__).parent.parent.parent / 'config.yaml'
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return Config.from_dict(config_dict)
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}. Using default configuration.")
        return Config()
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        print("Using default configuration.")
        return Config()


# Load configuration
config = load_config() 