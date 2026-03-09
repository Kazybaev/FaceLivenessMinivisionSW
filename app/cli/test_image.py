"""CLI: test a single image for liveness.

Replaces: test.py
Usage: python -m app.cli.test_image --image_name image_F1.jpg
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from app.infrastructure.config import get_settings
from app.infrastructure.logging_setup import setup_logging
from app.adapters.detectors.retinaface_detector import RetinaFaceDetector
from app.adapters.preprocessors.opencv_preprocessor import OpenCVPreprocessor
from app.adapters.repositories.filesystem_model_repo import FilesystemModelRepo
from app.ml.utils import parse_model_name
from app.ml.data.transforms import Compose, ToTensor

import torch
import torch.nn.functional as F


SAMPLE_IMAGE_PATH = "./images/sample/"


def check_image(image: np.ndarray) -> bool:
    height, width, channel = image.shape
    if width / height != 3 / 4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    return True


def test(image_name: str, model_dir: str, device_id: int) -> None:
    setup_logging()
    settings = get_settings()

    detector = RetinaFaceDetector(settings.detector.retinaface)
    preprocessor = OpenCVPreprocessor()
    model_repo = FilesystemModelRepo(model_dir)
    transform = Compose([ToTensor()])
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    image = cv2.imread(SAMPLE_IMAGE_PATH + image_name)
    result = check_image(image)
    if result is False:
        return

    face = detector.detect(image)
    if face is None:
        print("No face detected!")
        return
    image_bbox = face.bbox

    prediction = np.zeros((1, 3))
    test_speed = 0

    model_infos = model_repo.list_models()
    for info in model_infos:
        model = model_repo.load_model(info)
        model.to(device)
        model.eval()

        if info.scale is not None:
            img = preprocessor.crop_face(image, image_bbox, info.scale, info.w_input, info.h_input)
        else:
            img = cv2.resize(image, (info.w_input, info.h_input))

        img_tensor = transform(img).unsqueeze(0).to(device)
        start = time.time()
        with torch.inference_mode():
            result = model(img_tensor)
            result = F.softmax(result, dim=1).cpu().numpy()
        prediction += result
        test_speed += time.time() - start

    label = np.argmax(prediction)
    value = prediction[0][label] / len(model_infos)

    if label == 1:
        print(f"Image '{image_name}' is Real Face. Score: {value:.2f}.")
        result_text = f"RealFace Score: {value:.2f}"
        color = (255, 0, 0)
    else:
        print(f"Image '{image_name}' is Fake Face. Score: {value:.2f}.")
        result_text = f"FakeFace Score: {value:.2f}"
        color = (0, 0, 255)

    print(f"Prediction cost {test_speed:.2f} s")

    cv2.rectangle(
        image,
        (image_bbox.x, image_bbox.y),
        (image_bbox.x2, image_bbox.y2),
        color, 2,
    )
    cv2.putText(
        image, result_text,
        (image_bbox.x, image_bbox.y - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5 * image.shape[0] / 1024, color,
    )

    format_ = os.path.splitext(image_name)[-1]
    result_image_name = image_name.replace(format_, "_result" + format_)
    cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test single image for liveness")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default="./resources/anti_spoof_models")
    parser.add_argument("--image_name", type=str, default="image_F1.jpg")
    args = parser.parse_args()
    test(args.image_name, args.model_dir, args.device_id)
