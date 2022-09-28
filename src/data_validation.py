import os
from PIL import Image
from config import ValidateDatasetConfig


def validate_data(config: ValidateDatasetConfig):
    data_root = config.dataset_path
    images_path = os.path.join(data_root, 'images')
    labels_path = os.path.join(data_root, 'labels')
    for dataset_type in os.listdir(images_path):
        for image_filename in os.listdir(os.path.join(images_path, dataset_type)):
            Image.open(os.path.join(images_path, dataset_type, image_filename))
            if not os.path.exists(os.path.join(labels_path, dataset_type, image_filename.replace('.jpeg', '.txt'))):
                raise FileNotFoundError(f"label file not exist for {image_filename}")


if __name__ == '__main__':
    config = ValidateDatasetConfig.parse_raw()
    validate_data(config)