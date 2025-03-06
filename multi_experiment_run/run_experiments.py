from pathlib import Path
import yaml
import subprocess
import sys
import argparse
from typing import Dict, Any

# Add project root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

def deep_merge(base: dict, update: dict) -> dict:
    """Deep merge two dictionaries, with update values taking precedence."""
    result = base.copy()
    for key, value in update.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load full configuration from yaml file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def update_config(experiment: dict, base_config: dict):
    """Update config.yaml with experiment configuration merged with base config."""
    config_path = PROJECT_ROOT / 'config.yaml'
    
    # Deep merge base config with experiment-specific settings
    config = deep_merge(base_config, {
        'training': experiment.get('training', {}),
        'model': experiment.get('model', {})
    })
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

def run_experiments(config_path: Path):
    """Run all experiments defined in the config file."""
    # Load full configuration
    full_config = load_config(config_path)
    
    # Extract base config and experiments list
    base_config = full_config.get('base_config', {})
    experiments = full_config.get('experiments', [])
    
    if not experiments:
        print("Error: No experiments defined in config file")
        sys.exit(1)
    
    if not base_config:
        print("Warning: No base configuration found. Proceeding with experiment-specific configs only.")
    
    for experiment in experiments:
        print(f"\nRunning experiment: {experiment['name']}")
        print("="*50)
        
        # Update config.yaml with experiment settings merged with base config
        update_config(experiment, base_config)
        
        # Run train.py
        train_script = PROJECT_ROOT / 'src' / 'train.py'
        subprocess.run([sys.executable, str(train_script)], check=True)
        
        print(f"\nCompleted experiment: {experiment['name']}")
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Run a set of experiments defined in a YAML config file.')
    parser.add_argument('--config', 
                      default='multi_config.yaml',
                      help='Path to experiments config file (default: multi_config.yaml)')
    
    args = parser.parse_args()
    
    # If path is relative, make it relative to script directory
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = SCRIPT_DIR / config_path
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
        
    run_experiments(config_path)

if __name__ == "__main__":
    main() 