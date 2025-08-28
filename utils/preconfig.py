import copy
import json
from typing import Any

## Derived from Huggingface transformers
class PreConfig:
    def dict_dtype_to_str(self, d: dict[str, Any]) -> None:
        if d.get("dtype") is not None:
            if isinstance(d["dtype"], dict):
                d["dtype"] = {k: str(v).split(".")[-1] for k, v in d["dtype"].items()}
            elif not isinstance(d["dtype"], str):
                d["dtype"] = str(d["dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self.dict_dtype_to_str(value)

    def to_dict(self) -> dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type

        self.dict_dtype_to_str(output)

        return output
    
    def to_json_string(self) -> str:
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def __iter__(self):
        yield from self.__dict__