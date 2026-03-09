# -*- coding: utf-8 -*-
# Real-time liveness detection demo with webcam

import os
import sys
import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F

# Добавляем корень проекта в путь
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
from src.data_io import transform as trans
from src.utility import get_kernel, parse_model_name

MODEL_DIR = os.path.join(ROOT, "resources", "anti_spoof_models")
DETECT_MODEL = os.path.join(ROOT, "resources", "detection_model")

CONFIDENCE_THRESHOLD = 0.6
SMOOTHING_FRAMES = 10

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE
}


class FaceDetector:
    def __init__(self):
        caffemodel = os.path.join(DETECT_MODEL, "Widerface-RetinaFace.caffemodel")
        deploy = os.path.join(DETECT_MODEL, "deploy.prototxt")
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.confidence_thresh = 0.6

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))),
                             interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        if out[max_conf_index, 2] < self.confidence_thresh:
            return None
        left   = out[max_conf_index, 3] * width
        top    = out[max_conf_index, 4] * height
        right  = out[max_conf_index, 5] * width
        bottom = out[max_conf_index, 6] * height
        return [int(left), int(top), int(right - left + 1), int(bottom - top + 1)]


class LivenessModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self._load_all_models()

    def _load_all_models(self):
        for model_name in os.listdir(MODEL_DIR):
            if not model_name.endswith(".pth"):
                continue
            model_path = os.path.join(MODEL_DIR, model_name)
            h_input, w_input, model_type, _ = parse_model_name(model_name)
            kernel_size = get_kernel(h_input, w_input)
            model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)
            keys = iter(state_dict)
            first_key = next(keys)
            if 'module.' in first_key:
                from collections import OrderedDict
                new_sd = OrderedDict()
                for k, v in state_dict.items():
                    new_sd[k[7:]] = v
                model.load_state_dict(new_sd)
            else:
                model.load_state_dict(state_dict)
            model.eval()
            self.models[model_name] = (model, h_input, w_input)
        print(f"Загружено моделей: {len(self.models)}")

    def predict(self, face):
        transform = trans.Compose([trans.ToTensor()])
        real_scores = []
        fake_scores = []

        for model_name, (model, h_input, w_input) in self.models.items():
            face_resized = cv2.resize(face, (w_input, h_input))
            img = transform(face_resized).unsqueeze(0).to(self.device)
            with torch.no_grad():
                result = model(img)
                result = F.softmax(result, dim=1).cpu().numpy()
            # [0] = fake, [1] = real, [2] = neutral (игнорируем)
            fake_scores.append(float(result[0][0]))
            real_scores.append(float(result[0][1]))

        avg_real = np.mean(real_scores)
        avg_fake = np.mean(fake_scores)
        total = avg_real + avg_fake
        if total == 0:
            return None, None
        return avg_real / total, avg_fake / total


def main():
    print("Инициализация детектора лиц...")
    detector = FaceDetector()
    print("Загрузка моделей liveness...")
    liveness = LivenessModel()
    print("Запуск камеры...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Камера не найдена! Попробуйте изменить 0 на 1 или 2")
        return

    print("Готово! Нажмите Q для выхода.")
    score_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        try:
            bbox = detector.get_bbox(frame)

            if bbox is None:
                cv2.putText(display, "Лицо не обнаружено", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            else:
                x, y, w, h = bbox
                x  = max(0, x)
                y  = max(0, y)
                x2 = min(frame.shape[1], x + w)
                y2 = min(frame.shape[0], y + h)
                face = frame[y:y2, x:x2]

                if face.size > 0 and w >= 30 and h >= 30:
                    real_prob, fake_prob = liveness.predict(face)

                    if real_prob is not None:
                        # Сглаживание
                        score_history.append(real_prob)
                        if len(score_history) > SMOOTHING_FRAMES:
                            score_history.pop(0)
                        smoothed = np.mean(score_history)

                        is_real = smoothed >= 0.5
                        confidence = smoothed if is_real else (1 - smoothed)
                        label = "REAL" if is_real else "FAKE"
                        color = (0, 220, 0) if is_real else (0, 0, 220)

                        if confidence < CONFIDENCE_THRESHOLD:
                            label += " ?"

                        # Рамка
                        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                        # Текст
                        text = f"{label}  {confidence*100:.0f}%"
                        cv2.putText(display, text, (x, max(y - 12, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
                        # Полоса уверенности
                        bar_w = int(w * confidence)
                        cv2.rectangle(display, (x, y+h+4), (x+bar_w, y+h+14), color, -1)
                        cv2.rectangle(display, (x, y+h+4), (x+w, y+h+14), (180,180,180), 1)

        except Exception as e:
            cv2.putText(display, f"Ошибка: {str(e)[:40]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        cv2.imshow("Liveness Detection  |  Q - выход", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()