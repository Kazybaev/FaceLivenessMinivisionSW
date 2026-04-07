import numpy as np

from app.utils.preprocess import preprocess_face_crop


def test_preprocess_face_crop_returns_expected_tensor_shape() -> None:
    crop = np.full((180, 160, 3), 127, dtype=np.uint8)
    tensor = preprocess_face_crop(
        crop,
        input_size=128,
        mean=(0.5931, 0.4690, 0.4229),
        std=(0.2471, 0.2214, 0.2157),
    )
    assert tensor.shape == (1, 3, 128, 128)
    assert tensor.dtype == np.float32
