# Day 4 - Model Architecture Implementation

## Implemented

- `src/models/chest_disease_model.py`
- `SimpleChestDiseaseModel`
- Frozen VGG16 backbone
- `GlobalAveragePooling2D`
- `Dense(128, relu)`
- `Dropout(0.3)`
- `Dense(2, softmax)`
- Adam optimizer with learning rate `1e-3`
- Sparse categorical cross-entropy
- `print_model_insights()` parameter and shape report
- Unit tests for architecture and compilation
- `MODEL_ARCHITECTURE.md` design justification

## Testing note

The test suite constructs VGG16 with `weights=None` so CI does not depend on a large external model-weight download. The production/training default remains `weights="imagenet"`.

## Commands

```bash
python -m pytest -q
```

Build the intended baseline in Python:

```python
from src.models.chest_disease_model import SimpleChestDiseaseModel, print_model_insights

model = SimpleChestDiseaseModel().build(weights="imagenet")
print_model_insights(model)
model.summary()
```

## Day 4 boundary

This day establishes the baseline architecture only. No accuracy or clinical performance claim is made here. Training and medical evaluation come later.
