from typing import Any

from torchmetrics import F1Score, MetricCollection, Precision, Recall


def get_metrics(**kwargs: Any) -> MetricCollection:
    return MetricCollection(
        {
            'f1': F1Score(**kwargs),
            'precision': Precision(**kwargs),
            'recall': Recall(**kwargs),
        },
    )
