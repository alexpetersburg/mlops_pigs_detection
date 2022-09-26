from argparse import Namespace
from pathlib import Path

from clearml import Task, TaskTypes, Dataset
from config import TrainDatasetConfig
from yolo5.train import parse_opt, main as train



def main(config: TrainDatasetConfig):
    opt = parse_opt()
    opt = Namespace(**{**vars(opt), **config.dict()})
    task: Task = Task.init(project_name=config.project,
                           task_name=config.task_name,
                           task_type=TaskTypes.data_processing,
                           output_uri=True)

    clearml_params = {
        "dataset_id": config.dataset_id
    }
    task.connect(opt)
    dataset_path = Dataset.get(**clearml_params).get_local_copy()
    config.dataset_path = Path(dataset_path)
    train(opt=opt)


if __name__ == '__main__':
    config = TrainDatasetConfig.parse_raw()
    main(config=config)
