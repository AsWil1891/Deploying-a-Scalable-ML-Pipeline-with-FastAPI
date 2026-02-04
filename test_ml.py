import pytest
# TODO: add necessary import
import numpy as np
from sklearn.linear_model import LogisticRegression

from ml.model import compute_model_metrics, inference, train_model


# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """
    Test that compute_model_metrics returns expected precision/recall/F1
    for a simple known example.
    """
    y = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    # precision = 1/(1+0) = 1
    # recall = 1/(1+1) = 0.5
    # f1 = 2PR/(P+R) = 2*1*0.5/(1+0.5) = 0.666666...
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(2 * 1.0 * 0.5 / (1.0 + 0.5))


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    Test that train_model returns a LogisticRegression model (expected algorithm).
    """
    # Small deterministic dataset (AND gate-ish)
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 0, 0, 1])

    model = train_model(X_train, y_train)
    assert isinstance(model, LogisticRegression)


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    Test that inference returns predictions with correct length and as a numpy array.
    """
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 0, 0, 1])
    model = train_model(X_train, y_train)

    X_test = np.array([[0, 0], [1, 1]])
    preds = inference(model, X_test)

    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]
