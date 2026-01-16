from pydantic import BaseModel
from typing import Dict, Optional
import tomllib
import os

# --- Pydantic Models ---

class APIURL(BaseModel):
    UPDATE_DB: str

class APIConfig(BaseModel):
    HOST: str
    PORT: int
    URL: APIURL
    ABORT_SIGNAL: str

class LiteralTags(BaseModel):
    BLACKLIST_TAG: str
    EMPTY_TAG: str
    NO_TAGS_TAG: str

class Database(BaseModel):
    LITERAL_TAGS: LiteralTags

class CategorizerModel(BaseModel):
    max_output_length: int

class Models(BaseModel):
    Categorizer0: CategorizerModel

class TorchConfig(BaseModel):
    hardware_acceleration: str

class HardwareConfig(BaseModel):
    torch: TorchConfig

class Params(BaseModel):
    MODELS: Models
    HARDWARE: HardwareConfig

class AppConfig(BaseModel):
    API: APIConfig
    DB: Database
    PARAMS: Params

# --- Loader Function ---

def load_config(path: Optional[str] = None) -> AppConfig:
    """Load the TOML config file into an AppConfig object."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), '../../config/config.toml')
    with open(path, 'rb') as f:
        data = tomllib.load(f)
    return AppConfig.model_validate(data)

# --- Singleton Pattern for Shared Config ---

_config_instance: Optional[AppConfig] = None

def get_config() -> AppConfig:
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config()
    return _config_instance
