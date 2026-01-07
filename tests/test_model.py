import pickle
import numpy as np
import os


def test_model_file_exists():
    assert os.path.exists("models/model.pkl")


def test_scaler_file_exists():
    assert os.path.exists("models/scaler.pkl")


def test_model_prediction():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Create a dummy input with correct feature count
    sample_input = np.random.rand(1, scaler.mean_.shape[0])
    scaled_input = scaler.transform(sample_input)

    prediction = model.predict(scaled_input)

    assert prediction.shape == (1,)
    assert isinstance(prediction[0], (float, int, np.number))
