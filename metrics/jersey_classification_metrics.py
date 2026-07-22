import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

class JerseyClassificationMetrics:
    """
    Calculates Accuracy, F1-score, Precision, Recall, and Confusion Matrix for Jersey/Team classification.
    """
    def evaluate(self, y_true, y_pred, labels=None):
        """
        Args:
            y_true: list/array of true team or jersey class labels
            y_pred: list/array of predicted team or jersey class labels
            labels: optional list of label names
        Returns:
            dict containing Accuracy, F1-score (macro/weighted), Precision, Recall, confusion_matrix.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

        cm = confusion_matrix(y_true, y_pred)

        return {
            "Accuracy": float(acc),
            "F1-score": float(f1_macro),
            "F1-score (Weighted)": float(f1_weighted),
            "Precision": float(prec_macro),
            "Recall": float(rec_macro),
            "confusion_matrix": cm.tolist(),
            "unique_labels": labels if labels else np.unique(np.concatenate([y_true, y_pred])).tolist()
        }

    def plot_confusion_matrix(self, results, save_path=None):
        """
        Plots the confusion matrix heatmap for jersey classification.
        """
        cm = np.array(results.get("confusion_matrix", [[1, 0], [0, 1]]))
        class_labels = [str(lbl) for lbl in results.get("unique_labels", range(cm.shape[0]))]

        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
        fig.colorbar(cax)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=f"{cm[i, j]}", va="center", ha="center", size=14, fontweight="bold")

        ax.set_xticks(range(len(class_labels)))
        ax.set_yticks(range(len(class_labels)))
        ax.set_xticklabels(class_labels, fontsize=11)
        ax.set_yticklabels(class_labels, fontsize=11)
        ax.set_xlabel("Predicted Jersey / Team Label", fontsize=12)
        ax.set_ylabel("True Jersey / Team Label", fontsize=12)
        ax.set_title(f"Jersey Classification (Acc: {results['Accuracy']:.3f}, F1: {results['F1-score']:.3f})", fontsize=13, fontweight="bold")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        return fig
