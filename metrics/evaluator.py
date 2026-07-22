import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from .object_detection_metrics import ObjectDetectionMetrics
from .player_tracking_metrics import PlayerTrackingMetrics
from .ball_tracking_metrics import BallTrackingMetrics
from .jersey_classification_metrics import JerseyClassificationMetrics
from .possession_metrics import PossessionMetrics
from .camera_compensation_metrics import CameraCompensationMetrics
from .perspective_mapping_metrics import PerspectiveMappingMetrics
from .speed_estimation_metrics import SpeedEstimationMetrics
from .runtime_metrics import RuntimeTracker

class FootballAnalysisEvaluator:
    """
    Unified Evaluation System for Football Analysis Pipeline.
    Runs and aggregates evaluation across all 9 computer vision modules.
    """
    def __init__(self):
        self.results = {}
        self.raw_data = {}

    def evaluate_all(self, data_dict):
        """
        Runs evaluation for provided component data.
        """
        self.raw_data = data_dict

        if "object_detection" in data_dict:
            od_eval = ObjectDetectionMetrics()
            d = data_dict["object_detection"]
            self.results["Object Detection"] = od_eval.evaluate(d["preds"], d["gts"])

        if "player_tracking" in data_dict:
            pt_eval = PlayerTrackingMetrics()
            d = data_dict["player_tracking"]
            self.results["Player Tracking"] = pt_eval.evaluate(d["preds"], d["gts"])

        if "ball_tracking" in data_dict:
            bt_eval = BallTrackingMetrics()
            d = data_dict["ball_tracking"]
            self.results["Ball Tracking"] = bt_eval.evaluate(d["preds"], d["gts"])

        if "jersey_classification" in data_dict:
            jc_eval = JerseyClassificationMetrics()
            d = data_dict["jersey_classification"]
            self.results["Jersey Classification"] = jc_eval.evaluate(d["y_true"], d["y_pred"])

        if "possession" in data_dict:
            pos_eval = PossessionMetrics()
            d = data_dict["possession"]
            self.results["Possession"] = pos_eval.evaluate(d["pred_possession"], d["gt_possession"])

        if "camera_compensation" in data_dict:
            cc_eval = CameraCompensationMetrics()
            d = data_dict["camera_compensation"]
            self.results["Camera Compensation"] = cc_eval.evaluate(
                d["camera_movement_per_frame"], d.get("background_flow_vectors")
            )

        if "perspective_mapping" in data_dict:
            pm_eval = PerspectiveMappingMetrics()
            d = data_dict["perspective_mapping"]
            self.results["Perspective Mapping"] = pm_eval.evaluate(d["pred_pts"], d["gt_pts"])

        if "speed_estimation" in data_dict:
            se_eval = SpeedEstimationMetrics()
            d = data_dict["speed_estimation"]
            self.results["Speed Estimation"] = se_eval.evaluate(
                d["pred_speeds"], d["gt_speeds"], d.get("pred_distances"), d.get("gt_distances")
            )

        if "runtime" in data_dict:
            r = data_dict["runtime"]
            if "tracker" in r and isinstance(r["tracker"], RuntimeTracker):
                self.results["Runtime"] = r["tracker"].get_summary()
            elif isinstance(r, dict):
                self.results["Runtime"] = r

        return self.results

    def generate_markdown_report(self):
        """
        Generates formatted Markdown table report matching requested specifications.
        """
        rows = self._build_report_rows()
        df = pd.DataFrame(rows)
        md_table = df.to_markdown(index=False)
        return f"# Football Analysis Evaluation Summary\n\n{md_table}\n"

    def _build_report_rows(self):
        rows = []
        # 1. Object Detection
        od = self.results.get("Object Detection", {})
        od_str = f"mAP50: {od.get('mAP50', 0):.3f} | mAP50-95: {od.get('mAP50-95', 0):.3f} | Precision: {od.get('Precision', 0):.3f} | Recall: {od.get('Recall', 0):.3f}" if od else "N/A"
        rows.append({"Component": "Object Detection", "Metric to Report": od_str})

        # 2. Player Tracking
        pt = self.results.get("Player Tracking", {})
        pt_str = f"IDF1: {pt.get('IDF1', 0):.3f} | MOTA: {pt.get('MOTA', 0):.3f} | ID switches: {pt.get('ID_switches', 0)}" if pt else "N/A"
        rows.append({"Component": "Player Tracking", "Metric to Report": pt_str})

        # 3. Ball Tracking
        bt = self.results.get("Ball Tracking", {})
        bt_str = f"Precision: {bt.get('Precision', 0):.3f} | Recall: {bt.get('Recall', 0):.3f} | Detection rate: {bt.get('detection_rate', 0):.3f}" if bt else "N/A"
        rows.append({"Component": "Ball Tracking", "Metric to Report": bt_str})

        # 4. Jersey Classification
        jc = self.results.get("Jersey Classification", {})
        jc_str = f"Accuracy: {jc.get('Accuracy', 0):.3f} | F1-score: {jc.get('F1-score', 0):.3f}" if jc else "N/A"
        rows.append({"Component": "Jersey Classification", "Metric to Report": jc_str})

        # 5. Possession
        pos = self.results.get("Possession", {})
        pos_str = f"Accuracy against GT: {pos.get('Accuracy', 0):.3f} ({pos.get('Accuracy', 0)*100:.1f}%)" if pos else "N/A"
        rows.append({"Component": "Possession", "Metric to Report": pos_str})

        # 6. Camera Compensation
        cc = self.results.get("Camera Compensation", {})
        cc_str = f"Before: {cc.get('mean_displacement_before', 0):.2f} px -> After: {cc.get('mean_displacement_after', 0):.2f} px ({cc.get('jitter_reduction_pct', 0):.1f}% reduction)" if cc else "N/A"
        rows.append({"Component": "Camera Compensation", "Metric to Report": cc_str})

        # 7. Perspective Mapping
        pm = self.results.get("Perspective Mapping", {})
        pm_str = f"MAE: {pm.get('MAE_meters', 0):.2f} m | RMSE: {pm.get('RMSE_meters', 0):.2f} m | Max Error: {pm.get('Max_Error_meters', 0):.2f} m" if pm else "N/A"
        rows.append({"Component": "Perspective Mapping", "Metric to Report": pm_str})

        # 8. Speed Estimation
        se = self.results.get("Speed Estimation", {})
        se_str = f"Speed MAE: {se.get('Speed_MAE_kmh', 0):.2f} km/h | Error: {se.get('Relative_Speed_Error_pct', 0):.1f}%" if se else "N/A"
        rows.append({"Component": "Speed Estimation", "Metric to Report": se_str})

        # 9. Runtime
        rt = self.results.get("Runtime", {})
        rt_str = f"FPS: {rt.get('FPS', 0):.1f} | Latency: {rt.get('latency_mean_ms', 0):.1f} ms | CPU: {rt.get('CPU_usage_pct', 0):.1f}% | RAM: {rt.get('RAM_usage_mb', 0):.0f} MB | GPU: {rt.get('GPU_device', 'N/A')} ({rt.get('GPU_memory_mb', 0):.0f} MB)" if rt else "N/A"
        rows.append({"Component": "Runtime", "Metric to Report": rt_str})

        return rows

    def export_summary(self, json_path=None, csv_path=None):
        """
        Exports metrics results to JSON or CSV file.
        """
        if json_path:
            with open(json_path, "w") as f:
                json.dump(self.results, f, indent=4, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
        if csv_path:
            df = pd.DataFrame([{"Component": k, "Metrics": str(v)} for k, v in self.results.items()])
            df.to_csv(csv_path, index=False)

    def plot_summary_table_image(self, save_path="metrics_table.jpg"):
        """
        Renders the summary metrics table as a high-resolution JPG image card.
        """
        rows = self._build_report_rows()
        headers = ["Component", "Metric to Report"]
        cell_text = [[r["Component"], r["Metric to Report"]] for r in rows]

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis("off")
        ax.set_title("Football Analysis System - Component Metrics Summary", fontsize=16, fontweight="bold", pad=20)

        table = ax.table(cellText=cell_text, colLabels=headers, loc="center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

        # Style table headers and alternating row colors
        for (row_idx, col_idx), cell in table.get_celld().items():
            if row_idx == 0:
                cell.set_facecolor("#1f77b4")
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            else:
                if row_idx % 2 == 0:
                    cell.set_facecolor("#f2f2f2")
                else:
                    cell.set_facecolor("#ffffff")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, format="jpg", bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_dashboard(self, save_path="metrics_dashboard.jpg"):
        """
        Creates a high-level visual dashboard summarizing performance across components.
        """
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        fig.suptitle("Football Analysis Pipeline Evaluation Dashboard", fontsize=16, fontweight="bold", y=0.98)

        # 1. Object Detection
        od = self.results.get("Object Detection", {})
        axes[0, 0].bar(["mAP50", "mAP50-95", "Prec", "Rec"], [od.get("mAP50", 0), od.get("mAP50-95", 0), od.get("Precision", 0), od.get("Recall", 0)], color="#1f77b4")
        axes[0, 0].set_ylim(0, 1.1)
        axes[0, 0].set_title("Object Detection", fontweight="bold")
        axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)

        # 2. Player Tracking
        pt = self.results.get("Player Tracking", {})
        axes[0, 1].bar(["IDF1", "MOTA", "IDP", "IDR"], [pt.get("IDF1", 0), pt.get("MOTA", 0), pt.get("IDP", 0), pt.get("IDR", 0)], color="#2ca02c")
        axes[0, 1].set_ylim(0, 1.1)
        axes[0, 1].set_title(f"Player Tracking (IDSW={pt.get('ID_switches', 0)})", fontweight="bold")
        axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5)

        # 3. Ball Tracking
        bt = self.results.get("Ball Tracking", {})
        axes[0, 2].bar(["Prec", "Rec", "DetRate"], [bt.get("Precision", 0), bt.get("Recall", 0), bt.get("detection_rate", 0)], color="#ff7f0e")
        axes[0, 2].set_ylim(0, 1.1)
        axes[0, 2].set_title("Ball Tracking", fontweight="bold")
        axes[0, 2].grid(axis="y", linestyle="--", alpha=0.5)

        # 4. Jersey Classification
        jc = self.results.get("Jersey Classification", {})
        axes[1, 0].bar(["Accuracy", "F1-Score"], [jc.get("Accuracy", 0), jc.get("F1-score", 0)], color="#9467bd")
        axes[1, 0].set_ylim(0, 1.1)
        axes[1, 0].set_title("Jersey Classification", fontweight="bold")
        axes[1, 0].grid(axis="y", linestyle="--", alpha=0.5)

        # 5. Possession
        pos = self.results.get("Possession", {})
        axes[1, 1].bar(["Accuracy"], [pos.get("Accuracy", 0)], color="#8c564b")
        axes[1, 1].set_ylim(0, 1.1)
        axes[1, 1].set_title("Possession Accuracy", fontweight="bold")
        axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)

        # 6. Camera Compensation
        cc = self.results.get("Camera Compensation", {})
        axes[1, 2].bar(["Before (px)", "After (px)"], [cc.get("mean_displacement_before", 0), cc.get("mean_displacement_after", 0)], color=["#d62728", "#2ca02c"])
        axes[1, 2].set_title(f"Camera Movement Jitter", fontweight="bold")
        axes[1, 2].grid(axis="y", linestyle="--", alpha=0.5)

        # 7. Perspective Mapping
        pm = self.results.get("Perspective Mapping", {})
        axes[2, 0].bar(["MAE (m)", "RMSE (m)"], [pm.get("MAE_meters", 0), pm.get("RMSE_meters", 0)], color="#e377c2")
        axes[2, 0].set_title("Pitch Mapping Error", fontweight="bold")
        axes[2, 0].grid(axis="y", linestyle="--", alpha=0.5)

        # 8. Speed Estimation
        se = self.results.get("Speed Estimation", {})
        axes[2, 1].bar(["MAE (km/h)", "RMSE (km/h)"], [se.get("Speed_MAE_kmh", 0), se.get("Speed_RMSE_kmh", 0)], color="#bcbd22")
        axes[2, 1].set_title("Speed Estimation Error", fontweight="bold")
        axes[2, 1].grid(axis="y", linestyle="--", alpha=0.5)

        # 9. Runtime
        rt = self.results.get("Runtime", {})
        axes[2, 2].bar(["FPS", "Latency (ms)", "CPU (%)"], [rt.get("FPS", 0), rt.get("latency_mean_ms", 0), rt.get("CPU_usage_pct", 0)], color="#17becf")
        axes[2, 2].set_title("Runtime Performance", fontweight="bold")
        axes[2, 2].grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            plt.savefig(save_path, dpi=300, format="jpg", bbox_inches="tight")
        plt.close(fig)
        return save_path

    def save_all_visual_results(self, output_dir="output_metrics"):
        """
        Generates and saves all component visual plots + summary dashboard + table card as JPG files.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []

        # 1. Summary Table Image JPG
        table_path = os.path.join(output_dir, "metrics_summary_table.jpg")
        self.plot_summary_table_image(table_path)
        saved_files.append(table_path)

        # 2. Overall Dashboard JPG
        dash_path = os.path.join(output_dir, "metrics_dashboard.jpg")
        self.plot_dashboard(dash_path)
        saved_files.append(dash_path)

        # 3. Object Detection PR Curve JPG
        if "Object Detection" in self.results:
            od_eval = ObjectDetectionMetrics()
            p = os.path.join(output_dir, "object_detection_pr_curve.jpg")
            fig = od_eval.plot_precision_recall_curve(self.results["Object Detection"], save_path=p)
            plt.close(fig)
            saved_files.append(p)

        # 4. Player Tracking Summary JPG
        if "Player Tracking" in self.results:
            pt_eval = PlayerTrackingMetrics()
            p = os.path.join(output_dir, "player_tracking_summary.jpg")
            fig = pt_eval.plot_tracking_summary(self.results["Player Tracking"], save_path=p)
            plt.close(fig)
            saved_files.append(p)

        # 5. Ball Tracking Timeline JPG
        if "Ball Tracking" in self.results:
            bt_eval = BallTrackingMetrics()
            p = os.path.join(output_dir, "ball_tracking_timeline.jpg")
            fig = bt_eval.plot_ball_detection_timeline(self.results["Ball Tracking"], save_path=p)
            plt.close(fig)
            saved_files.append(p)

        # 6. Jersey Classification Confusion Matrix JPG
        if "Jersey Classification" in self.results:
            jc_eval = JerseyClassificationMetrics()
            p = os.path.join(output_dir, "jersey_classification_confusion_matrix.jpg")
            fig = jc_eval.plot_confusion_matrix(self.results["Jersey Classification"], save_path=p)
            plt.close(fig)
            saved_files.append(p)

        # 7. Possession Timeline JPG
        if "Possession" in self.results:
            pos_eval = PossessionMetrics()
            p = os.path.join(output_dir, "possession_timeline.jpg")
            pred_pos = self.raw_data.get("possession", {}).get("pred_possession")
            gt_pos = self.raw_data.get("possession", {}).get("gt_possession")
            fig = pos_eval.plot_possession_timeline(self.results["Possession"], pred_possession=pred_pos, gt_possession=gt_pos, save_path=p)
            plt.close(fig)
            saved_files.append(p)

        # 8. Camera Compensation Displacement JPG
        if "Camera Compensation" in self.results:
            cc_eval = CameraCompensationMetrics()
            p = os.path.join(output_dir, "camera_compensation_displacement.jpg")
            fig = cc_eval.plot_displacement_comparison(self.results["Camera Compensation"], save_path=p)
            plt.close(fig)
            saved_files.append(p)

        # 9. Perspective Mapping Error Map JPG
        if "Perspective Mapping" in self.results:
            pm_eval = PerspectiveMappingMetrics()
            p = os.path.join(output_dir, "perspective_mapping_error_map.jpg")
            fig = pm_eval.plot_pitch_error_map(self.results["Perspective Mapping"], save_path=p)
            plt.close(fig)
            saved_files.append(p)

        # 10. Speed Estimation Analysis JPG
        if "Speed Estimation" in self.results:
            se_eval = SpeedEstimationMetrics()
            p = os.path.join(output_dir, "speed_estimation_analysis.jpg")
            fig = se_eval.plot_speed_error_analysis(self.results["Speed Estimation"], save_path=p)
            plt.close(fig)
            saved_files.append(p)

        # 11. Runtime Summary JPG
        if "Runtime" in self.results:
            rt_tracker = RuntimeTracker()
            p = os.path.join(output_dir, "runtime_summary.jpg")
            fig = rt_tracker.plot_runtime_summary(self.results["Runtime"], save_path=p)
            plt.close(fig)
            saved_files.append(p)

        return saved_files
