from typing import Union
from pathlib import Path
from pydantic_yaml import YamlModel


class AddDatasetConfig(YamlModel):
    # data
    dataset_path: str
    dataset_name: str
    dataset_project: str

    @classmethod
    def parse_raw(cls, filename: Union[str, Path] = "add_data_config.yaml", *args, **kwargs):
        with open(filename, 'r') as f:
            data = f.read()
        return super().parse_raw(data, *args, **kwargs)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class ValidateDatasetConfig(YamlModel):
    # data
    dataset_path: str
    dataset_name: str
    project: str
    task_name: str
    dataset_id: str

    @classmethod
    def parse_raw(cls, filename: Union[str, Path] = "validate_dataset_config.yaml", *args, **kwargs):
        with open(filename, 'r') as f:
            data = f.read()
        return super().parse_raw(data, *args, **kwargs)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)