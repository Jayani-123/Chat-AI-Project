# config.py
import os
import yaml
from dotenv import load_dotenv


def load_config(path: str = "config.yml") -> dict:
    """
    Load environment variables from .env, then load YAML configuration.
    Ensures all API keys and config values are accessible across modules.

    Args:
        path (str): Path to the config.yml file.
    Returns:
        dict: Merged configuration dictionary.
    """
    # Load .env file
    load_dotenv()

    # Read YAML config file
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Recursively resolve env vars like ${VAR} in YAML
    def resolve_env_vars(obj):
        if isinstance(obj, dict):
            return {k: resolve_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            var_name = obj[2:-1]
            resolved = os.getenv(var_name)
            if resolved is None:
                print(f" Warning: Env var '{var_name}' not found in .env")
                return obj  # Fallback to original
            return resolved
        return obj

    config = resolve_env_vars(config)

    # Ensure required API keys are available (post-resolution)
    google_key = config.get('api_keys', {}).get('google_api_key')
    if not google_key:
        print("Warning: GOOGLE_API_KEY not found in .env or config.yml")

    owm_key = config.get('api_keys', {}).get('openweather_api_key')
    if not owm_key:
        print("Warning: OPENWEATHERMAP_API_KEY not found in .env or config.yml")
    else:
        # Make sure it's visible to OpenWeatherMap wrapper
        os.environ["OPENWEATHERMAP_API_KEY"] = owm_key

    return config


# Optional helper to fetch nested keys safely
def get_config_value(config: dict, key_path: str, default=None):
    """
    Get a nested configuration value using dot notation.
    Example: get_config_value(cfg, "models.chat.name")
    """
    keys = key_path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


# Global config instance (this exports 'config' for imports)
config = load_config()