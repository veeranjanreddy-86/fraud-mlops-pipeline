import numpy as np

from fraud_mlops.metrics import evaluate, recall_at_precision, threshold_for_cost, threshold_for_precision

Y = np.array([0, 0, 0, 0, 1, 0, 1, 1])
S = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9])


def test_recall_at_precision():
    # Flagging >=0.7 gives precision 1.0, recall 2/3; >=0.5 gives precision 0.75, recall 1.0.
    assert recall_at_precision(Y, S, 1.0) == 2 / 3
    assert recall_at_precision(Y, S, 0.75) == 1.0


def test_threshold_for_precision_maximises_recall():
    assert threshold_for_precision(Y, S, 1.0) == 0.7
    assert threshold_for_precision(Y, S, 0.75) == 0.5


def test_threshold_for_cost():
    # Missing fraud is expensive -> flag everything from 0.5 up.
    assert threshold_for_cost(Y, S, fn_cost=100, fp_cost=1) == 0.5
    # False positives expensive -> only the clean top band.
    assert threshold_for_cost(Y, S, fn_cost=1, fp_cost=100) == 0.7


def test_evaluate_confusion_counts():
    m = evaluate(Y, S, threshold=0.5, target_precision=0.75)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 1, 0, 4)
    assert m["precision"] == 0.75 and m["recall"] == 1.0
