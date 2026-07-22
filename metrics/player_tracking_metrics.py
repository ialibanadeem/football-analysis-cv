import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0.0

class PlayerTrackingMetrics:
    """
    Calculates MOTA, IDF1, and ID Switches (IDSW) for multi-object player tracking.
    """
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

    def evaluate(self, pred_tracks_per_frame, gt_tracks_per_frame):
        """
        Args:
            pred_tracks_per_frame: list of dicts {track_id: bbox [x1,y1,x2,y2]} per frame
            gt_tracks_per_frame: list of dicts {gt_id: bbox [x1,y1,x2,y2]} per frame
        Returns:
            dict containing MOTA, IDF1, IDSW, IDP, IDR, TP, FP, FN.
        """
        total_gt = 0
        total_fp = 0
        total_fn = 0
        total_idsw = 0

        last_gt_to_pred_map = {}
        frame_stats = []

        # Collect global identity counts for IDF1
        gt_ids_set = set()
        pred_ids_set = set()

        for gts in gt_tracks_per_frame:
            gt_ids_set.update(gts.keys())
        for preds in pred_tracks_per_frame:
            pred_ids_set.update(preds.keys())

        # Global matching matrix for ID metrics
        id_tp_matrix = {gt_id: {pred_id: 0 for pred_id in pred_ids_set} for gt_id in gt_ids_set}

        for frame_idx, (preds, gts) in enumerate(zip(pred_tracks_per_frame, gt_tracks_per_frame)):
            num_gt = len(gts)
            total_gt += num_gt

            gt_ids = list(gts.keys())
            pred_ids = list(preds.keys())

            if len(gt_ids) == 0:
                total_fp += len(pred_ids)
                frame_stats.append({"fp": len(pred_ids), "fn": 0, "idsw": 0})
                continue

            if len(pred_ids) == 0:
                total_fn += num_gt
                frame_stats.append({"fp": 0, "fn": num_gt, "idsw": 0})
                continue

            # Compute IoU matrix
            iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))
            for i, gid in enumerate(gt_ids):
                for j, pid in enumerate(pred_ids):
                    iou_matrix[i, j] = calculate_iou(gts[gid], preds[pid])

            # Cost matrix for Hungarian algorithm (minimize 1 - IoU)
            cost_matrix = 1.0 - iou_matrix
            gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

            current_matches = {}
            frame_fp = 0
            frame_fn = 0
            frame_idsw = 0

            matched_gt_set = set()
            matched_pred_set = set()

            for g_idx, p_idx in zip(gt_indices, pred_indices):
                if iou_matrix[g_idx, p_idx] >= self.iou_threshold:
                    gid = gt_ids[g_idx]
                    pid = pred_ids[p_idx]
                    matched_gt_set.add(gid)
                    matched_pred_set.add(pid)
                    current_matches[gid] = pid
                    id_tp_matrix[gid][pid] += 1

                    # Check for ID Switch
                    if gid in last_gt_to_pred_map and last_gt_to_pred_map[gid] != pid:
                        frame_idsw += 1
                    last_gt_to_pred_map[gid] = pid

            frame_fn = len(gt_ids) - len(matched_gt_set)
            frame_fp = len(pred_ids) - len(matched_pred_set)

            total_fn += frame_fn
            total_fp += frame_fp
            total_idsw += frame_idsw

            frame_stats.append({"fp": frame_fp, "fn": frame_fn, "idsw": frame_idsw})

        # Calculate MOTA
        mota = 1.0 - (total_fp + total_fn + total_idsw) / (total_gt + 1e-10)

        # Calculate IDF1 using global optimal bipartite matching on overlaps
        gt_list = list(gt_ids_set)
        pred_list = list(pred_ids_set)

        if len(gt_list) > 0 and len(pred_list) > 0:
            overlap_matrix = np.zeros((len(gt_list), len(pred_list)))
            for i, gid in enumerate(gt_list):
                for j, pid in enumerate(pred_list):
                    overlap_matrix[i, j] = id_tp_matrix[gid][pid]

            # Maximize overlap (minimize negative overlap)
            row_ind, col_ind = linear_sum_assignment(-overlap_matrix)
            id_tp = overlap_matrix[row_ind, col_ind].sum()
        else:
            id_tp = 0.0

        id_fp = max(0, sum(len(p) for p in pred_tracks_per_frame) - id_tp)
        id_fn = max(0, total_gt - id_tp)

        idp = id_tp / (id_tp + id_fp + 1e-10)
        idr = id_tp / (id_tp + id_fn + 1e-10)
        idf1 = 2 * id_tp / (2 * id_tp + id_fp + id_fn + 1e-10)

        return {
            "IDF1": float(idf1),
            "MOTA": float(mota),
            "ID_switches": int(total_idsw),
            "IDP": float(idp),
            "IDR": float(idr),
            "FP": int(total_fp),
            "FN": int(total_fn),
            "Total_GT": int(total_gt),
            "frame_stats": frame_stats
        }

    def plot_tracking_summary(self, results, save_path=None):
        """
        Plots tracking metrics summary and per-frame error distribution.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart of main tracking metrics
        metrics = ["IDF1", "MOTA", "IDP", "IDR"]
        values = [results.get(m, 0.0) for m in metrics]
        colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

        axes[0].bar(metrics, values, color=colors, alpha=0.85)
        axes[0].set_ylim(0, 1.1)
        for i, v in enumerate(values):
            axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
        axes[0].set_title("Player Tracking Overview Metrics", fontsize=13, fontweight="bold")
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)

        # Line plot of frame-by-frame errors
        stats = results.get("frame_stats", [])
        fps = [s["fp"] for s in stats]
        fns = [s["fn"] for s in stats]
        idsws = [s["idsw"] for s in stats]
        frames = range(len(stats))

        axes[1].plot(frames, fps, label="False Positives (FP)", color="#ff7f0e", alpha=0.7)
        axes[1].plot(frames, fns, label="False Negatives (FN)", color="#d62728", alpha=0.7)
        axes[1].plot(frames, idsws, label="ID Switches (IDSW)", color="#9467bd", linewidth=2)
        axes[1].set_xlabel("Frame Number", fontsize=11)
        axes[1].set_ylabel("Count", fontsize=11)
        axes[1].set_title(f"Per-Frame Errors (Total IDSW = {results.get('ID_switches', 0)})", fontsize=13, fontweight="bold")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        return fig
