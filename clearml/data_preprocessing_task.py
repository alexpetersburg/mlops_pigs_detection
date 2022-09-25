from pathlib import Path

from clearml import Task, TaskTypes, Dataset
from config import PreprocessDatasetConfig
from src.data_preprocessing import preprocess_data


def main(config: PreprocessDatasetConfig):
    task = Task.init(project_name=config.project,
                     task_name=config.task_name, task_type=TaskTypes.data_processing)
    clearml_params = {
        "dataset_id": config.dataset_id
    }
    task.connect(config)
    dataset_path = Dataset.get(**clearml_params).get_local_copy()
    config.dataset_path = Path(dataset_path)
    preprocess_data(config=config)
    dataset = Dataset.create(dataset_project=config.project, dataset_name=config.output_dataset_name)
    dataset.add_files(config.output_dataset_path)
    task.set_parameter("output_dataset_id", dataset.id)
    dataset.upload()
    dataset.finalize()


if __name__ == '__main__':
    config = PreprocessDatasetConfig.parse_raw()
    main(config=config)
