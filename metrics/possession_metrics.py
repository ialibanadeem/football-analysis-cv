import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

class PossessionMetrics:
    """
    Calculates Possession Accuracy and possession breakdown error against ground truth.
    """
    def evaluate(self, pred_possession, gt_possession, team_names=None):
        """
        Args:
            pred_possession: array/list of frame-by-frame predicted team IDs (e.g. [1, 1, 2, 0, ...])
            gt_possession: array/list of frame-by-frame true team IDs
            team_names: optional dict mapping ID -> name (e.g. {1: "Team 1", 2: "Team 2", 0: "Loose/None"})
        Returns:
            dict containing Accuracy, team possession breakdown errors, and confusion matrix.
        """
        pred_possession = np.array(pred_possession)
        gt_possession = np.array(gt_possession)

        acc = accuracy_score(gt_possession, pred_possession)
        cm = confusion_matrix(gt_possession, pred_possession)

        unique_teams = np.unique(np.concatenate([gt_possession, pred_possession]))

        pred_distribution = {int(t): float(np.mean(pred_possession == t) * 100) for t in unique_teams}
        gt_distribution = {int(t): float(np.mean(gt_possession == t) * 100) for t in unique_teams}

        distribution_error = {
            int(t): float(abs(pred_distribution[t] - gt_distribution[t])) for t in unique_teams
        }

        return {
            "Accuracy": float(acc),
            "predicted_distribution_pct": pred_distribution,
            "gt_distribution_pct": gt_distribution,
            "distribution_error_pct": distribution_error,
            "confusion_matrix": cm.tolist(),
            "teams": unique_teams.tolist(),
            "team_names": team_names or {int(t): f"Team {t}" if t != 0 else "None" for t in unique_teams}
        }

    def plot_possession_timeline(self, results, pred_possession=None, gt_possession=None, save_path=None):
        """
        Plots possession team comparison and frame-level timeline.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

        teams = results.get("teams", [1, 2, 0])
        team_names = results.get("team_names", {1: "Team 1", 2: "Team 2", 0: "None"})
        labels = [team_names.get(int(t), str(t)) for t in teams]

        pred_pct = [results["predicted_distribution_pct"].get(int(t), 0.0) for t in teams]
        gt_pct = [results["gt_distribution_pct"].get(int(t), 0.0) for t in teams]

        x = np.arange(len(labels))
        width = 0.35

        axes[0].bar(x - width/2, gt_pct, width, label="Ground Truth", color="#1f77b4", alpha=0.85)
        axes[0].bar(x + width/2, pred_pct, width, label="Predicted", color="#ff7f0e", alpha=0.85)
        axes[0].set_ylabel("Possession Share (%)", fontsize=11)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, fontsize=11)
        axes[0].set_title(f"Possession Distribution (Overall Accuracy: {results['Accuracy']:.3f})", fontsize=13, fontweight="bold")
        axes[0].legend(loc="upper right")
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)

        if pred_possession is not None and gt_possession is not None:
            frames = range(len(pred_possession))
            axes[1].plot(frames, gt_possession, label="GT Possession", color="#1f77b4", alpha=0.7, linewidth=2)
            axes[1].plot(frames, pred_possession, label="Predicted Possession", color="#ff7f0e", alpha=0.7, linestyle="--")
            axes[1].set_yticks(teams)
            axes[1].set_yticklabels(labels)
            axes[1].set_xlabel("Frame Number", fontsize=11)
            axes[1].set_title("Frame-by-Frame Possession Sequence", fontsize=13, fontweight="bold")
            axes[1].legend(loc="upper right")
            axes[1].grid(True, linestyle="--", alpha=0.5)
        else:
            axes[1].axis("off")
            axes[1].text(0.5, 0.5, "Pass pred_possession & gt_possession arrays\nto view timeline plot.", ha="center", va="center", fontsize=12)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        return fig
