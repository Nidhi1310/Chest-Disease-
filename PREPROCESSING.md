# Preprocessing Contract

## Pipeline
Raw chest X-ray -> load -> resize to 224 x 224 -> VGG16-specific preprocessing -> batch -> model.

## VGG16 input contract
- Shape: (224, 224, 3)
- Use TensorFlow/Keras VGG16 `preprocess_input`
- Do not substitute generic `/255.0` normalization for the VGG16 ImageNet baseline.

## Consistency
Training, validation, testing, and inference must use the same preprocessing implementation.

Day 2 will implement the reusable preprocessing module and validation tests.
