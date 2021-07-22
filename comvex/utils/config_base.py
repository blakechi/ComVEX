import json


class ConfigBase(object):
    @classmethod
    def from_json(cls, config_path: str) -> "ConfigBase":
        with open(config_path, "r") as f:
            config = json.load(f)

        return cls(**config)

    def __len__(self) -> int:
        return self.__class__.__dict__.__len__()

    def __str__(self) -> str:
        return f"{self.__class__.__name__} = {json.dumps(self.__dict__, indent=4)}"