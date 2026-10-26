import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class ModelConfig:
    id: str
    path: str
    max_model_len: int
    reasoning: bool
    gpu_memory_utilization: float
    max_num_seqs: int


@dataclass
class VLLMConfig:
    host: str
    port: int
    dtype: str
    tensor_parallel_size: int
    enable_prefix_caching: bool


@dataclass
class GenConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 32768
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None


class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict:
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: config.yaml not found at {self.config_path}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)

    def get_model_config(self, model_id: str) -> ModelConfig:
        if model_id not in self.config["models"]:
            raise ValueError(f"Model '{model_id}' not found in config")
        model = self.config["models"][model_id]
        return ModelConfig(
            id=model_id,
            path=model["path"],
            max_model_len=model["max_model_len"],
            reasoning=model.get("reasoning", False),
            gpu_memory_utilization=model["gpu_memory_utilization"],
            max_num_seqs=model["max_num_seqs"],
        )

    def get_vllm_config(self) -> VLLMConfig:
        vllm = self.config["vllm"]
        return VLLMConfig(
            host=vllm["host"],
            port=vllm["port"],
            dtype=vllm["dtype"],
            tensor_parallel_size=vllm["tensor_parallel_size"],
            enable_prefix_caching=vllm["enable_prefix_caching"],
        )

    def get_gen_config(self) -> GenConfig:
        gen = self.config.get("generation", {})
        return GenConfig(
            temperature=gen.get("temperature", 0.7),
            top_p=gen.get("top_p", 0.9),
            max_tokens=gen.get("max_tokens", 2048),
            frequency_penalty=gen.get("frequency_penalty", 0.0),
            presence_penalty=gen.get("presence_penalty", 0.0),
        )
