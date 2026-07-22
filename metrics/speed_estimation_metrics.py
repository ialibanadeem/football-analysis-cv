import numpy as np
import matplotlib.pyplot as plt

class SpeedEstimationMetrics:
    """
    Calculates Error against known/reference distance and speed (MAE, RMSE, Relative % Error).
    """
    def evaluate(self, pred_speeds, gt_speeds, pred_distances=None, gt_distances=None):
        """
        Args:
            pred_speeds: list/array of estimated speeds in km/h
            gt_speeds: list/array of ground truth reference speeds in km/h
            pred_distances: optional list/array of estimated cumulative distances in meters
            gt_distances: optional list/array of ground truth reference distances in meters
        Returns:
            dict containing Speed MAE, Speed RMSE, Distance MAE, Distance RMSE, and percentage errors.
        """
        pred_sp = np.array(pred_speeds)
        gt_sp = np.array(gt_speeds)

        speed_diff = pred_sp - gt_sp
        speed_mae = float(np.mean(np.abs(speed_diff)))
        speed_rmse = float(np.sqrt(np.mean(speed_diff**2)))
        relative_speed_error_pct = float(np.mean(np.abs(speed_diff) / (np.abs(gt_sp) + 1e-5)) * 100)

        results = {
            "Speed_MAE_kmh": speed_mae,
            "Speed_RMSE_kmh": speed_rmse,
            "Relative_Speed_Error_pct": relative_speed_error_pct,
            "pred_speeds": pred_sp.tolist(),
            "gt_speeds": gt_sp.tolist()
        }

        if pred_distances is not None and gt_distances is not None:
            pred_dist = np.array(pred_distances)
            gt_dist = np.array(gt_distances)
            dist_diff = pred_dist - gt_dist
            dist_mae = float(np.mean(np.abs(dist_diff)))
            dist_rmse = float(np.sqrt(np.mean(dist_diff**2)))
            relative_dist_error_pct = float(np.mean(np.abs(dist_diff) / (np.abs(gt_dist) + 1e-5)) * 100)

            results.update({
                "Distance_MAE_m": dist_mae,
                "Distance_RMSE_m": dist_rmse,
                "Relative_Distance_Error_pct": relative_dist_error_pct,
                "pred_distances": pred_dist.tolist(),
                "gt_distances": gt_dist.tolist()
            })

        return results

    def plot_speed_error_analysis(self, results, save_path=None):
        """
        Plots predicted vs reference speed curves and scatter correlation plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        pred_sp = np.array(results.get("pred_speeds", []))
        gt_sp = np.array(results.get("gt_speeds", []))
        samples = range(len(pred_sp))

        # Time series comparison
        axes[0].plot(samples, gt_sp, label="Ground Truth Reference", color="#1f77b4", linewidth=2)
        axes[0].plot(samples, pred_sp, label="Estimated Speed", color="#ff7f0e", linestyle="--", linewidth=1.8)
        axes[0].set_xlabel("Sample Index / Frame Window", fontsize=11)
        axes[0].set_ylabel("Speed (km/h)", fontsize=11)
        axes[0].set_title(f"Speed Estimation (MAE: {results['Speed_MAE_kmh']:.2f} km/h)", fontsize=13, fontweight="bold")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Correlation scatter plot
        if len(pred_sp) > 0:
            axes[1].scatter(gt_sp, pred_sp, color="#2ca02c", alpha=0.7, edgecolors="k")
            max_val = max(np.max(gt_sp), np.max(pred_sp)) if len(gt_sp) > 0 else 30
            axes[1].plot([0, max_val], [0, max_val], 'r--', label="Ideal Identity (x=y)")
            axes[1].set_xlabel("Reference Speed (km/h)", fontsize=11)
            axes[1].set_ylabel("Estimated Speed (km/h)", fontsize=11)
            axes[1].set_title(f"Reference vs Estimated Speed (Error: {results['Relative_Speed_Error_pct']:.1f}%)", fontsize=13, fontweight="bold")
            axes[1].legend(loc="upper left")
            axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        return fig
