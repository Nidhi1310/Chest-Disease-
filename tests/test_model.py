import tensorflow as tf

from src.models.chest_disease_model import SimpleChestDiseaseModel, print_model_insights


def build_test_model():
    """Build without downloading ImageNet weights."""
    return SimpleChestDiseaseModel().build(weights=None)


def test_model_input_and_output_shapes():
    model = build_test_model()

    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 2)


def test_vgg16_backbone_is_frozen():
    model = build_test_model()
    backbone = model.layers[0]

    assert backbone.name.startswith("vgg16")
    assert backbone.trainable is False
    assert all(layer.trainable is False for layer in backbone.layers)


def test_classification_head_matches_day4_design():
    model = build_test_model()
    layer_types = [type(layer).__name__ for layer in model.layers]

    assert layer_types == [
        "Functional",
        "GlobalAveragePooling2D",
        "Dense",
        "Dropout",
        "Dense",
    ]
    assert model.layers[2].units == 128
    assert model.layers[2].activation.__name__ == "relu"
    assert model.layers[3].rate == 0.3
    assert model.layers[4].units == 2
    assert model.layers[4].activation.__name__ == "softmax"


def test_only_classification_head_is_trainable():
    model = build_test_model()

    trainable_names = {weight.name for weight in model.trainable_weights}
    assert trainable_names
    assert all("vgg16" not in name.lower() for name in trainable_names)


def test_model_is_compiled_for_sparse_binary_classification():
    model = build_test_model()

    assert model.loss == "sparse_categorical_crossentropy"
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
    assert abs(float(model.optimizer.learning_rate.numpy()) - 1e-3) < 1e-6


def test_model_insights_report_parameter_breakdown():
    model = build_test_model()
    insights = print_model_insights(model)

    assert insights["total_params"] > 0
    assert insights["trainable_params"] > 0
    assert insights["non_trainable_params"] > insights["trainable_params"]
    assert insights["input_shape"] == (None, 224, 224, 3)
    assert insights["output_shape"] == (None, 2)
