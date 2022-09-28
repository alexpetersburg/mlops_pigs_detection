import os
import shutil

import numpy as np
import cv2
from config import PreprocessDatasetConfig


def preprocess_data(config: PreprocessDatasetConfig):
    mask = cv2.imread(config.mask_path, cv2.IMREAD_GRAYSCALE) > 100
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    images_path = os.path.join(config.dataset_path, 'images')
    labels_path = os.path.join(config.dataset_path, 'labels')

    shutil.rmtree(config.output_dataset_path)
    output_images_path = os.path.join(config.output_dataset_path, 'images')
    output_labels_path = os.path.join(config.output_dataset_path, 'labels')

    for dataset_type in os.listdir(images_path):
        os.makedirs(os.path.join(output_images_path, dataset_type), exist_ok=True)
        os.makedirs(os.path.join(output_labels_path, dataset_type), exist_ok=True)
        for image_filename in os.listdir(os.path.join(images_path, dataset_type)):
            image = cv2.imread(os.path.join(images_path, dataset_type, image_filename))
            masked_image = (image * mask).astype(np.uint8)
            cv2.imwrite(os.path.join(output_images_path, dataset_type, image_filename), masked_image)

            label_filename = image_filename.replace('.jpeg', '.txt')
            shutil.copyfile(os.path.join(labels_path, dataset_type, label_filename),
                            os.path.join(output_labels_path, dataset_type, label_filename))


if __name__ == '__main__':
    config = PreprocessDatasetConfig.parse_raw()
    preprocess_data(config)