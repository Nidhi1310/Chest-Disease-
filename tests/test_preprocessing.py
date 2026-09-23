import numpy as np
from PIL import Image

from src.preprocessing import VGG16Preprocessor


def test_vgg16_preprocessing_shape():
    image = np.zeros((224, 224, 3), dtype=np.float32)
    result = VGG16Preprocessor().preprocess_array(image)

    assert result.shape == (224, 224, 3)


def test_vgg16_preprocessing_matches_official_function():
    from tensorflow.keras.applications.vgg16 import preprocess_input

    image = np.full((224, 224, 3), 128.0, dtype=np.float32)

    expected = preprocess_input(np.array(image, copy=True))
    actual = VGG16Preprocessor().preprocess_array(image)

    np.testing.assert_allclose(actual, expected)


def test_preprocessing_is_deterministic():
    image = np.arange(224 * 224 * 3, dtype=np.float32).reshape(224, 224, 3)
    image %= 256

    preprocessor = VGG16Preprocessor()

    first = preprocessor.preprocess_array(image)
    second = preprocessor.preprocess_array(image)

    np.testing.assert_array_equal(first, second)


def test_preprocessing_is_not_simple_zero_to_one_normalization():
    image = np.full((224, 224, 3), 128.0, dtype=np.float32)

    result = VGG16Preprocessor().preprocess_array(image)
    simple_normalization = image / 255.0

    assert not np.allclose(result, simple_normalization)
    assert result.max() > 1.0
    assert result.min() >= -130.0


def test_image_loader_resizes_to_vgg16_shape(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (80, 40), color=(128, 128, 128)).save(image_path)

    result = VGG16Preprocessor().preprocess_image(image_path)

    assert result.shape == (224, 224, 3)


def test_invalid_raw_shape_is_rejected():
    bad_image = np.zeros((128, 128, 3), dtype=np.float32)

    try:
        VGG16Preprocessor().preprocess_array(bad_image)
    except ValueError as exc:
        assert "Expected image shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid shape")


def test_out_of_range_raw_values_are_rejected():
    bad_image = np.full((224, 224, 3), 300.0, dtype=np.float32)

    try:
        VGG16Preprocessor().preprocess_array(bad_image)
    except ValueError as exc:
        assert "[0, 255]" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid pixel range")
