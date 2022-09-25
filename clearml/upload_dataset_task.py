from clearml import Dataset
from config import AddDatasetConfig


def main(data_config: AddDatasetConfig):
    dataset = Dataset.create(
        dataset_name=data_config.dataset_name,
        dataset_project=data_config.dataset_project
    )

    dataset.add_files(path=data_config.dataset_path)
    dataset.upload()
    dataset.finalize()


if __name__ == '__main__':
    config = AddDatasetConfig.parse_raw()
    main(data_config=config)
