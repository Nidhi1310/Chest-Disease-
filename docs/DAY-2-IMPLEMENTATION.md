# Day 2 - VGG16 Preprocessing

## Goal

Build one reusable preprocessing path for training, validation, testing, and inference.

## Implemented

- RGB image loading
- Resize to 224 x 224
- VGG16 `preprocess_input`
- Input shape/range validation
- Deterministic behavior
- Preprocessing statistics
- Unit tests against TensorFlow's official VGG16 preprocessing

## Why not /255?

VGG16 pretrained ImageNet weights expect the preprocessing convention used during their original training. Generic /255 scaling changes the input distribution instead of matching that convention.

## Local environment note

For this project, use a TensorFlow-supported Python version. The current official installation documentation should be checked before creating the environment.

On Windows, after installing Python 3.13, recreate the environment with:

```cmd
deactivate
rmdir /s /q .venv
py -3.13 -m venv .venv
.venv\\Scripts\\activate
python -m pip install -r requirements.txt
python -m pytest -q
```

The `py -3.13` launcher command assumes Python 3.13 is installed and registered with the Windows Python launcher.

## Day 2 completion criterion

All preprocessing tests pass, and you can explain why VGG16 preprocessing is different from simple /255 normalization.
