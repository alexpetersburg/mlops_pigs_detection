from pathlib import Path

from clearml import Task, TaskTypes, Dataset
from config import ValidateDatasetConfig
from src.data_validation import validate_data


def main(config: ValidateDatasetConfig):
    task = Task.init(project_name=config.project,
                     task_name=config.task_name, task_type=TaskTypes.data_processing)
    clearml_params = {
        "dataset_id": config.dataset_id
    }
    task.connect(config)
    dataset_path = Dataset.get(**clearml_params).get_local_copy()
    config.dataset_path = Path(dataset_path)
    validate_data(config=config)


if __name__ == '__main__':
    config = ValidateDatasetConfig.parse_raw()
    main(config=config)
