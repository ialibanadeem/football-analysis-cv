import numpy as np
import matplotlib.pyplot as plt

def calculate_center_distance(box1, box2):
    c1 = [(box1[0] + box1[2]) / 2.0, (box1[1] + box1[3]) / 2.0]
    c2 = [(box2[0] + box2[2]) / 2.0, (box2[1] + box2[3]) / 2.0]
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

class BallTrackingMetrics:
    """
    Calculates Precision, Recall, and Detection Rate for ball tracking.
    """
    def __init__(self, max_distance_pixels=30.0):
        self.max_distance_pixels = max_distance_pixels

    def evaluate(self, pred_ball_per_frame, gt_ball_per_frame):
        """
        Args:
            pred_ball_per_frame: list of dicts/lists or None per frame [{'bbox': [x1,y1,x2,y2]}] or dict of bbox
            gt_ball_per_frame: list of dicts/lists or None per frame [{'bbox': [x1,y1,x2,y2]}] or dict of bbox
        Returns:
            dict containing Precision, Recall, detection_rate, TP, FP, FN.
        """
        tp = 0
        fp = 0
        fn = 0
        frames_with_gt_ball = 0
        frames_with_tp_ball = 0

        frame_timeline = []  # 0: FN/FP/Miss, 1: TP, -1: No GT & No Pred

        for pred, gt in zip(pred_ball_per_frame, gt_ball_per_frame):
            gt_bbox = self._extract_bbox(gt)
            pred_bbox = self._extract_bbox(pred)

            if gt_bbox is not None:
                frames_with_gt_ball += 1

            if gt_bbox is None and pred_bbox is None:
                frame_timeline.append(0)  # No ball present, correctly not detected
                continue
            elif gt_bbox is None and pred_bbox is not None:
                fp += 1
                frame_timeline.append(-1)  # False Positive
                continue
            elif gt_bbox is not None and pred_bbox is None:
                fn += 1
                frame_timeline.append(-2)  # False Negative (Missed)
                continue

            # Both exist -> measure distance
            dist = calculate_center_distance(pred_bbox, gt_bbox)
            if dist <= self.max_distance_pixels:
                tp += 1
                frames_with_tp_ball += 1
                frame_timeline.append(1)  # True Positive
            else:
                fp += 1
                fn += 1
                frame_timeline.append(-1)

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        detection_rate = frames_with_tp_ball / (frames_with_gt_ball + 1e-10)

        return {
            "Precision": float(precision),
            "Recall": float(recall),
            "detection_rate": float(detection_rate),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "total_gt_frames": int(frames_with_gt_ball),
            "frame_timeline": frame_timeline
        }

    def _extract_bbox(self, item):
        if item is None:
            return None
        if isinstance(item, dict):
            if "bbox" in item:
                return item["bbox"]
            elif 1 in item and isinstance(item[1], dict) and "bbox" in item[1]:
                return item[1]["bbox"]
        elif isinstance(item, (list, tuple)) and len(item) > 0:
            if isinstance(item[0], dict) and "bbox" in item[0]:
                return item[0]["bbox"]
            elif len(item) == 4 and all(isinstance(x, (int, float)) for x in item):
                return item
        return None

    def plot_ball_detection_timeline(self, results, save_path=None):
        """
        Plots ball tracking metrics bar chart and frame-by-frame detection timeline.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

        metrics = ["Precision", "Recall", "detection_rate"]
        values = [results.get(m, 0.0) for m in metrics]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        axes[0].bar(metrics, values, color=colors, alpha=0.85)
        axes[0].set_ylim(0, 1.1)
        for i, v in enumerate(values):
            axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
        axes[0].set_title("Ball Tracking Performance", fontsize=13, fontweight="bold")
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)

        timeline = results.get("frame_timeline", [])
        frames = range(len(timeline))
        colors_map = {1: "green", -1: "orange", -2: "red", 0: "gray"}
        point_colors = [colors_map.get(status, "blue") for status in timeline]

        axes[1].scatter(frames, timeline, c=point_colors, s=15, alpha=0.7)
        axes[1].set_yticks([1, 0, -1, -2])
        axes[1].set_yticklabels(["Match (TP)", "No Ball", "Extra (FP)", "Miss (FN)"])
        axes[1].set_xlabel("Frame Number", fontsize=11)
        axes[1].set_title("Frame-by-Frame Ball Detection Timeline", fontsize=13, fontweight="bold")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        return fig
