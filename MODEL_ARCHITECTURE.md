# Day 4 — Model Architecture

## Baseline decision

The project uses a deliberately simple **VGG16 transfer-learning baseline** for the first model iteration.

VGG16 is chosen for three reasons:

1. It is an interpretable baseline for transfer learning.
2. It is a well-known convolutional architecture for vision tasks.
3. It makes the transfer-learning workflow easy to explain and verify.

This is **not** a claim that VGG16 is optimal for medical imaging, state-of-the-art, or production-efficient. Future experiments may compare architectures such as EfficientNet, ResNet, or Vision Transformers after the baseline is validated.

## Architecture

```text
Input: 224 x 224 x 3
        |
        v
VGG16 ImageNet backbone
Frozen (not trainable)
        |
        v
GlobalAveragePooling2D
        |
        v
Dense(128, ReLU)
        |
        v
Dropout(0.3)
        |
        v
Dense(2, Softmax)
```

## Why this head is intentionally simple

- **GlobalAveragePooling2D instead of Flatten:** reduces the number of parameters and avoids a large fully connected representation.
- **One Dense layer:** provides a small classification head without adding unnecessary complexity.
- **Dropout(0.3):** adds regularization while keeping the baseline easy to interpret.
- **Frozen VGG16 backbone:** isolates the first experiment to the classification head and keeps training computationally simpler.

## Training configuration

- Input: `(224, 224, 3)`
- Backbone weights: ImageNet by default
- Backbone: frozen
- Optimizer: Adam
- Learning rate: `1e-3`
- Loss: `sparse_categorical_crossentropy`
- Metrics: accuracy for baseline training monitoring
- Output: 2-class softmax

Medical evaluation is intentionally handled separately in the later evaluation stage. Accuracy is not treated as the complete clinical evaluation.

## Model inspection

`print_model_insights(model)` reports:

- total parameters
- trainable parameters
- non-trainable parameters
- input shape
- output shape

This makes it explicit that the first experiment trains the classification head while the pretrained VGG16 features remain fixed.

## Current boundary

Day 4 creates the model architecture and tests it. It does **not** claim final performance and does **not** yet fine-tune the VGG16 backbone.
