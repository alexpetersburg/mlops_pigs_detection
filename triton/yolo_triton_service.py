import cv2
import numpy as np
import torch

from yolo5.utils.augmentations import letterbox
from yolo5.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from collections import namedtuple
from typing import List
import tritonclient.grpc as grpcclient


YoloDetection = namedtuple('YoloDetection', 'bbox score class_name')


class YoloTriton:

    def __init__(self, url, model_name, model_inputs, model_outputs, img_size=(640, 640), dataset_config_path=None,
                 conf_thres=0.25, iou_thres=0.45, max_det=1000):
        """"""

        self.img_size = img_size
        self.dataset_config_path = dataset_config_path
        self.client = grpcclient.InferenceServerClient(url=url, verbose=True)
        self.model_name = model_name
        self.model_inputs = model_inputs
        self.model_outputs = model_outputs

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

    def detect(self, image: np.ndarray) -> List[YoloDetection]:
        """

        :param image: BGR np.ndarray
        :return:
        """
        im = self._preprocess_img(image)
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference

        pred = self.triton_process_input(np.expand_dims(image, axis=0))

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, max_det=self.max_det)
        pred_boxes = []
        # Process predictions
        for det in pred:  # per image

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], image.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    xywh = torch.tensor(xyxy).tolist()
                    xywh[2] = xywh[2] - xywh[0]
                    xywh[3] = xywh[3] - xywh[1]
                    pred_boxes.append(YoloDetection(xywh, conf.item(), self.model.names[int(cls)]))
        return pred_boxes

    def _preprocess_img(self, image):
        # Padded resize
        image = letterbox(image, self.img_size, stride=32, auto=True)[0]

        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        return np.ascontiguousarray(image)

    def triton_process_input(self, image: np.ndarray):
        inputs = []
        inputs.append(grpcclient.InferInput(self.model_inputs, image.shape, "FP32"))
        inputs[0].set_data_from_numpy(image)

        outputs = []
        outputs.append(grpcclient.InferRequestedOutput(self.model_outputs))

        return self.client.infer(self.model_name, inputs, outputs=outputs).as_numpy(self.model_outputs)


if __name__ == '__main__':
    triton = YoloTriton(url="0.0.0.0:8001", model_name="yolov5s_onnx", model_inputs='images', model_outputs='output')
    test_img = cv2.imread('test.jpeg')
    dets = triton.detect(test_img)
    print(dets)
