from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def classification_report_metrics(y_true, y_pred, labels=None):
    """
    Возвращает classification report в виде dict.
    """
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )


def accuracy(y_true, y_pred) -> float:
    """
    Простая accuracy для end-to-end и baseline оценки.
    """
    return float(accuracy_score(y_true, y_pred))


def build_confusion_matrix(y_true, y_pred, labels=None):
    """
    Возвращает confusion matrix как numpy array.
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


def precision_at_k(all_relevant, all_predicted, k=5) -> float:
    """
    all_relevant: список коллекций релевантных doc_id для каждого запроса
    all_predicted: список списков предсказанных doc_id в порядке ранжирования
    """
    scores = []

    for relevant, predicted in zip(all_relevant, all_predicted):
        relevant_set = set(relevant)
        predicted_k = predicted[:k]
        hit_count = sum(1 for item in predicted_k if item in relevant_set)
        scores.append(hit_count / k)

    return float(np.mean(scores)) if scores else 0.0


def recall_at_k(all_relevant, all_predicted, k=5) -> float:
    scores = []

    for relevant, predicted in zip(all_relevant, all_predicted):
        relevant_set = set(relevant)
        if len(relevant_set) == 0:
            scores.append(0.0)
            continue

        predicted_k = predicted[:k]
        hit_count = sum(1 for item in predicted_k if item in relevant_set)
        scores.append(hit_count / len(relevant_set))

    return float(np.mean(scores)) if scores else 0.0


def mrr(all_relevant, all_predicted, k=5) -> float:
    scores = []

    for relevant, predicted in zip(all_relevant, all_predicted):
        relevant_set = set(relevant)
        rr = 0.0

        for rank, item in enumerate(predicted[:k], start=1):
            if item in relevant_set:
                rr = 1.0 / rank
                break

        scores.append(rr)

    return float(np.mean(scores)) if scores else 0.0