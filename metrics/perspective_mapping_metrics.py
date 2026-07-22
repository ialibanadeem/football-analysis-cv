import numpy as np
import matplotlib.pyplot as plt

class PerspectiveMappingMetrics:
    """
    Calculates Pixel-to-Pitch Coordinate Error (MAE, RMSE in meters) for Perspective Mapping.
    """
    def evaluate(self, predicted_pitch_pts, gt_pitch_pts):
        """
        Args:
            predicted_pitch_pts: list/array of transformed pitch points [[x_m, y_m], ...]
            gt_pitch_pts: list/array of true reference pitch points [[x_m, y_m], ...]
        Returns:
            dict containing MAE (meters), RMSE (meters), Max Error (meters), and per-point errors.
        """
        pred_pts = np.array([p for p in predicted_pitch_pts if p is not None])
        gt_pts = np.array([g for p, g in zip(predicted_pitch_pts, gt_pitch_pts) if p is not None])

        if len(pred_pts) == 0:
            return {
                "MAE_meters": 0.0,
                "RMSE_meters": 0.0,
                "Max_Error_meters": 0.0,
                "euclidean_errors": []
            }

        # Calculate Euclidean distance errors in meters
        diffs = pred_pts - gt_pts
        euclidean_errors = np.sqrt(np.sum(diffs**2, axis=1))

        mae = float(np.mean(euclidean_errors))
        rmse = float(np.sqrt(np.mean(euclidean_errors**2)))
        max_err = float(np.max(euclidean_errors))

        return {
            "MAE_meters": mae,
            "RMSE_meters": rmse,
            "Max_Error_meters": max_err,
            "euclidean_errors": euclidean_errors.tolist(),
            "pred_pts": pred_pts.tolist(),
            "gt_pts": gt_pts.tolist()
        }

    def plot_pitch_error_map(self, results, pitch_length=23.32, pitch_width=68.0, save_path=None):
        """
        Plots error distribution scatter map on top of a football pitch schematic.
        """
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Pitch Scatter Error Plot
        gt_pts = np.array(results.get("gt_pts", []))
        errors = np.array(results.get("euclidean_errors", []))

        if len(gt_pts) > 0:
            sc = axes[0].scatter(gt_pts[:, 0], gt_pts[:, 1], c=errors, cmap="YlOrRd", s=60, edgecolors="black")
            fig.colorbar(sc, ax=axes[0], label="Error (meters)")
        axes[0].set_xlim(-2, pitch_length + 2)
        axes[0].set_ylim(-2, pitch_width + 2)
        axes[0].set_xlabel("Pitch Length (m)", fontsize=11)
        axes[0].set_ylabel("Pitch Width (m)", fontsize=11)
        axes[0].set_title(f"Pitch Error Spatial Map (MAE = {results['MAE_meters']:.2f} m)", fontsize=13, fontweight="bold")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Histogram of Error Distribution
        if len(errors) > 0:
            axes[1].hist(errors, bins=15, color="#1f77b4", edgecolor="black", alpha=0.75)
            axes[1].axvline(results["MAE_meters"], color="red", linestyle="dashed", linewidth=2, label=f"MAE: {results['MAE_meters']:.2f} m")
            axes[1].axvline(results["RMSE_meters"], color="orange", linestyle="dashed", linewidth=2, label=f"RMSE: {results['RMSE_meters']:.2f} m")
            axes[1].set_xlabel("Euclidean Error (meters)", fontsize=11)
            axes[1].set_ylabel("Frequency", fontsize=11)
            axes[1].set_title("Coordinate Error Histogram", fontsize=13, fontweight="bold")
            axes[1].legend(loc="upper right")
            axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        return fig
