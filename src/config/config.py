from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

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


@dataclass
class VisualizationConfig:
    num_predictions: int = 32


@dataclass
class ModelConfig:
    name: str = "dual_pathway_fft"


@dataclass
class Config:
    training: TrainingConfig = TrainingConfig()
    signal: SignalConfig = SignalConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    model: ModelConfig = ModelConfig()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create a Config instance from a dictionary."""
        training_dict = config_dict.get('training', {})
        early_stopping_dict = training_dict.get('early_stopping', {})
        training_dict['early_stopping'] = EarlyStoppingConfig(**early_stopping_dict)
        
        return cls(
            training=TrainingConfig(**training_dict),
            signal=SignalConfig(**config_dict.get('signal', {})),
            visualization=VisualizationConfig(**config_dict.get('visualization', {})),
            model=ModelConfig(**config_dict.get('model', {}))
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