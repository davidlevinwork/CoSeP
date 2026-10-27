"""Evaluation metrics container."""
from dataclasses import dataclass, field


@dataclass
class Metrics:
    """Model evaluation metrics.

    Attributes:
        accuracy: Overall accuracy.
        precision: Macro-averaged precision.
        recall: Macro-averaged recall.
        f1_score: Macro-averaged F1 score.
        inference_time: Average time per batch (seconds).
        model_size: Size of the model's state_dict, in MB.
        per_class_metrics: For each class label, its precision, recall, F1 score and accuracy.
    """
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time: float
    model_size: float
    per_class_metrics: dict[int, dict[str, float]] = field(default_factory=dict)

    def __str__(self) -> str:
        per_class_str = "\n".join(
            f"Class {cls}: Precision: {m['precision']:.3%}, "
            f"Recall: {m['recall']:.3%}, F1: {m['f1_score']:.3%}"
            for cls, m in self.per_class_metrics.items()
        )
        return (
            f"Accuracy: {self.accuracy:.3%}, "
            f"Precision: {self.precision:.3%}, "
            f"Recall: {self.recall:.3%}, "
            f"F1: {self.f1_score:.3%}, "
            f"Inference Time: {self.inference_time:.4f}s, "
            f"Size: {self.model_size:.2f}MB\n"
            f"Per Class:\n{per_class_str}"
        )
