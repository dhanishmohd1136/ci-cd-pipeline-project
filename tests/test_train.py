
from src.train import train_model

def test_model_training():
    X_train, model, metrics = train_model("data/data.csv")

    assert model is not None
    assert len(X_train) > 0
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["f1"] <= 1