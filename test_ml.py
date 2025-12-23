import pandas as pd
import pytest
from ml.data import apply_label
import numpy as np
from ml.model import train_model
from ml.model import compute_model_metrics
# TODO: add necessary import

# TODO: implement the first test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    Verify `compute_model_metrics` for perfect and partial predictions.
    
    """
    #all metrics 1.0
    y = np.array([1, 0, 1, 0])
    preds = np.array([1, 0, 1, 0])
    p, r, f = compute_model_metrics(y, preds)
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(1.0)
    assert f == pytest.approx(1.0)

    # 1 true positive out of 2 predicted positives, precision=0.5
    # and 1 true positive out of 2 actual positives, recall=0.5, f1=0.5
    y2 = np.array([1, 0, 1, 0])
    preds2 = np.array([1, 1, 0, 0])
    p2, r2, f2 = compute_model_metrics(y2, preds2)
    assert p2 == pytest.approx(0.5)
    assert r2 == pytest.approx(0.5)
    assert f2 == pytest.approx(0.5)


# TODO: implement the second test. Change the function name and input as needed
def test_apply_label():
    """
    Test that `apply_label` converts binary predictions to strings.
    
    """
    # prediction encoded index 0 holds the class
    assert apply_label([1]) == ">50K"
    assert apply_label(np.array([0])) == "<=50K"



# TODO: implement the third test. Change the function name and input as needed
def test_train_model_trains_model():
    """
    Test that `train_model` fits a classifier that can predict labels.
    
    """
    # simple temp dataset (4 samples, 3 features)
    X_train = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]])
    y_train = np.array([0, 1, 0, 1])

    model = train_model(X_train, y_train)

    # model should expose a predict method
    assert hasattr(model, "predict")

    preds = model.predict(X_train)
    # predictions should have same shape as labels and contain binary values
    assert preds.shape == y_train.shape
    assert set(np.unique(preds)).issubset({0, 1})



