from pydantic import BaseModel
from typing import Dict, Optional
import tomllib
import os

# --- Pydantic Models ---


# --- API Endpoints Models ---
class EndpointActions(BaseModel):
    RUN: str
    ABORT: str
    UPDATE: str
    STATUS: str

class APIEndpointsGroup(BaseModel):
    UPDATE_DB_ROOT: str
    CATEGORIZER_ROOT: str
    SEARCH_ROOT: str
    UPDATE_DB: EndpointActions
    CATEGORIZER: EndpointActions

class APIConfig(BaseModel):
    HOST: str
    PORT: int
    ENDPOINTS: APIEndpointsGroup
    ABORT_SIGNAL: str  # TODO: move ABORT_SIGNAL to a specific level

class LiteralTags(BaseModel):
    BLACKLIST_TAG: str
    EMPTY_TAG: str
    NO_TAGS_TAG: str

class Database(BaseModel):
    LITERAL_TAGS: LiteralTags

class InferenceModel(BaseModel):
    max_output_length: int
    safety_input_length: int

class Models(BaseModel):
    default_model_config: InferenceModel
    Categorizer0: InferenceModel

class TorchConfig(BaseModel):
    hardware_acceleration: str

class HardwareConfig(BaseModel):
    torch: TorchConfig

class CategorizerConfig(BaseModel):
    STALL_SECONDS: int
    STALL_CHECK_FREQ: int
    NO_STALLING_ID: str

class Params(BaseModel):
    MODELS: Models
    HARDWARE: HardwareConfig
    CATEGORIZER: CategorizerConfig


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


if __name__ == "__main__":
    # For testing purposes
    config = load_config()
    print(config.model_dump_json(indent=4))