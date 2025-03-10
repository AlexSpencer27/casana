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
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()


@dataclass
class SignalConfig:
    length: int = 2048
    sampling_rate: int = 1024


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
    # Weights will be initialized from curriculum and updated during training
    position_weight: float = 100.0
    magnitude_weight: float = 1.0
    gradient_weight: float = 1e-8
    second_derivative_weight: float = 1e-4


@dataclass
class ComplexNoiseConfig:
    start_amplitude: float = 0.0
    end_amplitude: float = 0.2
    min_amplitude: float = 0.05
    epochs_to_max: int = 10000


@dataclass
class LossWeightsConfig:
    position: float = 100.0
    magnitude: float = 1.0
    gradient: float = 1e-8
    second_derivative: float = 1e-2


@dataclass
class LossWeightsCurriculumConfig:
    start: LossWeightsConfig = LossWeightsConfig()
    final: LossWeightsConfig = LossWeightsConfig(
        gradient=1e-6,
        second_derivative=1e-2
    )
    epochs_to_final: int = 20000


@dataclass
class LearningRateCurriculumConfig:
    start: float = 5e-5
    end: float = 1e-6
    epochs_to_min: int = 50000


@dataclass
class CurriculumConfig:
    complex_noise: ComplexNoiseConfig = ComplexNoiseConfig()
    learning_rate: LearningRateCurriculumConfig = LearningRateCurriculumConfig()
    loss_weights: LossWeightsCurriculumConfig = LossWeightsCurriculumConfig()


@dataclass
class Config:
    training: TrainingConfig = TrainingConfig()
    signal: SignalConfig = SignalConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    curriculum: CurriculumConfig = CurriculumConfig()

    def __post_init__(self):
        """Initialize loss weights from curriculum start values."""
        self.loss.position_weight = self.curriculum.loss_weights.start.position
        self.loss.magnitude_weight = self.curriculum.loss_weights.start.magnitude
        self.loss.gradient_weight = self.curriculum.loss_weights.start.gradient
        self.loss.second_derivative_weight = self.curriculum.loss_weights.start.second_derivative

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create a Config instance from a dictionary."""
        training_dict = config_dict.get('training', {})
        early_stopping_dict = training_dict.get('early_stopping', {})
        training_dict['early_stopping'] = EarlyStoppingConfig(**early_stopping_dict)
        
        model_dict = config_dict.get('model', {})
        gradient_refinement_dict = model_dict.get('gradient_refinement', {})
        model_dict['gradient_refinement'] = GradientRefinementConfig(**gradient_refinement_dict)
        
        curriculum_dict = config_dict.get('curriculum', {})
        if curriculum_dict:
            complex_noise_dict = curriculum_dict.get('complex_noise', {})
            curriculum_dict['complex_noise'] = ComplexNoiseConfig(**complex_noise_dict)
            
            lr_dict = curriculum_dict.get('learning_rate', {})
            curriculum_dict['learning_rate'] = LearningRateCurriculumConfig(**lr_dict)
            
            loss_weights_dict = curriculum_dict.get('loss_weights', {})
            if loss_weights_dict:
                start_weights_dict = loss_weights_dict.get('start', {})
                final_weights_dict = loss_weights_dict.get('final', {})
                loss_weights_dict['start'] = LossWeightsConfig(**start_weights_dict)
                loss_weights_dict['final'] = LossWeightsConfig(**final_weights_dict)
            curriculum_dict['loss_weights'] = LossWeightsCurriculumConfig(**loss_weights_dict)

        # Create config instance
        instance = cls(
            training=TrainingConfig(**training_dict),
            signal=SignalConfig(**config_dict.get('signal', {})),
            visualization=VisualizationConfig(**config_dict.get('visualization', {})),
            model=ModelConfig(**model_dict),
            loss=LossConfig(**config_dict.get('loss', {})),
            curriculum=CurriculumConfig(**curriculum_dict) if curriculum_dict else CurriculumConfig()
        )

        # Initialize loss weights from curriculum
        if curriculum_dict and 'loss_weights' in curriculum_dict:
            start_weights = instance.curriculum.loss_weights.start
            instance.loss.position_weight = start_weights.position
            instance.loss.magnitude_weight = start_weights.magnitude
            instance.loss.gradient_weight = start_weights.gradient
            instance.loss.second_derivative_weight = start_weights.second_derivative

        return instance


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