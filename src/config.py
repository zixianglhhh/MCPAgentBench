import json
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient

# load environment variables
load_dotenv()

api_key = os.getenv("API_KEY")

class ModelRegistry:
    """
    Manage multiple model clients:
    - Read {model, api_key, api_base/base_url, name} from configs/llm_config.json
    - If api_key is not provided, fall back to environment variable OPENAI_API_KEY
    - Expose a method to get a client by name/model
    """

    def __init__(self, config_path: str = "configs/llm_config.json") -> None:
        self._entries: List[Dict[str, Any]] = []
        self._name_to_entry: Dict[str, Dict[str, Any]] = {}
        self._model_to_entries: Dict[str, List[Dict[str, Any]]] = {}
        self._load(config_path)

    def _load(self, config_path: str) -> None:
        if not os.path.exists(config_path):
            entry = {
                "name": "openai_default",
                "model": "gpt-4o-mini",
                "client": OpenAIChatCompletionClient(
                    model="gpt-4o-mini",
                    api_key=api_key
                    ),
            }
            self._entries = [entry]
            self._name_to_entry[entry["name"]] = entry
            self._model_to_entries.setdefault(entry["model"], []).append(entry)
            return

        with open(config_path, "r", encoding="utf-8") as f:
            cfg_list = json.load(f)

        for idx, cfg in enumerate(cfg_list):
            model = cfg.get("model") or "gpt-4o-mini"
            base_url = cfg.get("api_base") or cfg.get("base_url")
            name_raw = cfg.get("name") or f"client_{idx}_{model}"
            model_info = cfg.get("model_info")
            api_key_env_name = cfg.get("api_key_env_name", "API_KEY")
            # ensure the name is a valid Python identifier
            name = name_raw.replace("-", "_").replace(" ", "_")
            if not (name[0].isalpha() or name[0] == "_"):
                name = f"m_{name}"

            # Get API key from the specified environment variable
            model_api_key = os.getenv(api_key_env_name)
            if not model_api_key:
                print(f"Warning: Environment variable {api_key_env_name} not set for model {model}, using default API_KEY")
                model_api_key = api_key

            client_kwargs = {"model": model, "api_key": model_api_key}
            
            if base_url:
                client_kwargs["base_url"] = base_url
            
            
            # Try to create client without model_info first
            try:
                entry = {
                    "name": name,
                    "model": model,
                    "client": OpenAIChatCompletionClient(**client_kwargs),
                }
                self._entries.append(entry)
                self._name_to_entry[name] = entry
                self._model_to_entries.setdefault(model, []).append(entry)
            except Exception as e:
                # If creation fails due to missing model_info, provide a minimal default
                if "model_info is required" in str(e):
                    # Use provided model_info if available, otherwise use minimal default
                    if not model_info:
                        model_info = {
                            "function_calling": True,
                            "json_output": True
                        }
                    client_kwargs["model_info"] = model_info
                    try:
                        entry = {
                            "name": name,
                            "model": model,
                            "client": OpenAIChatCompletionClient(**client_kwargs),
                        }
                        self._entries.append(entry)
                        self._name_to_entry[name] = entry
                        self._model_to_entries.setdefault(model, []).append(entry)
                    except Exception as e2:
                        print(f"Warning: Failed to create client for model {model}: {e2}")
                        continue
                else:
                    # Other errors, print warning and skip
                    print(f"Warning: Failed to create client for model {model}: {e}")
                    continue

        if not self._entries:
            entry = {
                "name": "openai_default",
                "model": "gpt-4o-mini",
                "client": OpenAIChatCompletionClient(
                    model="gpt-4o-mini", api_key=api_key
                ),
            }
            self._entries = [entry]
            self._name_to_entry[entry["name"]] = entry
            self._model_to_entries.setdefault(entry["model"], []).append(entry)

    def get(self, name_or_model: Optional[str] = None) -> Dict[str, Any]:
        if not name_or_model:
            return self._entries[0]
        if name_or_model in self._name_to_entry:
            return self._name_to_entry[name_or_model]
        if name_or_model in self._model_to_entries:
            return self._model_to_entries[name_or_model][0]
        return self._entries[0]
