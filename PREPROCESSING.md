# Preprocessing Contract

## Day 2 implementation

The preprocessing pipeline is now implemented in `src/preprocessing.py`.

## Pipeline

Raw chest X-ray -> load as RGB -> resize to 224 x 224 -> VGG16-specific `preprocess_input` -> model input.

## VGG16 input contract

- Shape: (224, 224, 3)
- Raw pixels: [0, 255]
- Preprocessing: `tensorflow.keras.applications.vgg16.preprocess_input`
- No generic `/255.0` normalization for the VGG16 ImageNet baseline.

TensorFlow's VGG16 preprocessing converts RGB input to BGR and zero-centers channels using the ImageNet channel means.

## Consistency

Training and inference must use the same `VGG16Preprocessor` implementation.

The Day 2 test suite verifies:
- expected output shape
- equality with the official TensorFlow preprocessing function
- deterministic output for the same input
- image resizing
- rejection of invalid shapes
- rejection of invalid raw pixel ranges

## Reusable API

`VGG16Preprocessor.preprocess_image(path)` loads and preprocesses an image.

`VGG16Preprocessor.preprocess_array(array)` validates and preprocesses an already resized RGB array.

`VGG16Preprocessor.stats(array)` returns shape, dtype, range, mean, and standard deviation for logging/debugging.
