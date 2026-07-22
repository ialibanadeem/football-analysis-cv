import numpy as np
import matplotlib.pyplot as plt

def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0.0

class ObjectDetectionMetrics:
    """
    Calculates mAP50, mAP50-95, Precision, and Recall for object detection.
    """
    def __init__(self, iou_thresholds=None):
        if iou_thresholds is None:
            self.iou_thresholds = np.linspace(0.50, 0.95, 10)
        else:
            self.iou_thresholds = np.array(iou_thresholds)

    def evaluate(self, predictions_per_frame, ground_truths_per_frame):
        """
        Args:
            predictions_per_frame: list of lists of dicts [{'bbox': [x1,y1,x2,y2], 'score': float, 'class': int}]
            ground_truths_per_frame: list of lists of dicts [{'bbox': [x1,y1,x2,y2], 'class': int}]
        Returns:
            dict containing mAP50, mAP50_95, precision, recall, and PR curve data.
        """
        all_aps = []
        ap50 = 0.0
        overall_tp50 = 0
        overall_fp50 = 0
        overall_fn50 = 0

        # Compute AP for each IoU threshold
        for iou_thresh in self.iou_thresholds:
            tp, fp, scores, total_gts = self._match_frame_predictions(
                predictions_per_frame, ground_truths_per_frame, iou_thresh
            )
            
            if total_gts == 0 or len(scores) == 0:
                ap = 0.0
            else:
                sort_idx = np.argsort(-scores)
                tp = tp[sort_idx]
                fp = fp[sort_idx]

                cum_tp = np.cumsum(tp)
                cum_fp = np.cumsum(fp)
                recalls = cum_tp / total_gts
                precisions = cum_tp / (cum_tp + cum_fp + 1e-10)

                # Compute 11-point interpolated AP
                ap = 0.0
                for r in np.linspace(0, 1, 11):
                    p_max = np.max(precisions[recalls >= r]) if np.any(recalls >= r) else 0.0
                    ap += p_max / 11.0

            all_aps.append(ap)

            if np.isclose(iou_thresh, 0.50):
                ap50 = ap
                overall_tp50 = int(np.sum(tp)) if len(tp) > 0 else 0
                overall_fp50 = int(np.sum(fp)) if len(fp) > 0 else 0
                overall_fn50 = total_gts - overall_tp50

        p50 = overall_tp50 / (overall_tp50 + overall_fp50 + 1e-10)
        r50 = overall_tp50 / (overall_tp50 + overall_fn50 + 1e-10)
        map50_95 = float(np.mean(all_aps))

        # Store PR curve for iou=0.50
        tp50, fp50, scores50, total_gts50 = self._match_frame_predictions(
            predictions_per_frame, ground_truths_per_frame, 0.50
        )
        pr_recalls, pr_precisions = self._compute_pr_curve(tp50, fp50, scores50, total_gts50)

        results = {
            "mAP50": float(ap50),
            "mAP50-95": map50_95,
            "Precision": float(p50),
            "Recall": float(r50),
            "pr_curve": {"recalls": pr_recalls, "precisions": pr_precisions}
        }
        return results

    def _match_frame_predictions(self, preds_frames, gts_frames, iou_thresh):
        tp_list = []
        fp_list = []
        score_list = []
        total_gts = 0

        for preds, gts in zip(preds_frames, gts_frames):
            total_gts += len(gts)
            gt_matched = [False] * len(gts)

            sorted_preds = sorted(preds, key=lambda x: x.get('score', 1.0), reverse=True)

            for pred in sorted_preds:
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, gt in enumerate(gts):
                    if gt_matched[gt_idx]:
                        continue
                    if pred.get('class', 0) != gt.get('class', 0):
                        continue
                    iou = calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_thresh and best_gt_idx >= 0:
                    tp_list.append(1)
                    fp_list.append(0)
                    gt_matched[best_gt_idx] = True
                else:
                    tp_list.append(0)
                    fp_list.append(1)

                score_list.append(pred.get('score', 1.0))

        return np.array(tp_list), np.array(fp_list), np.array(score_list), total_gts

    def _compute_pr_curve(self, tp, fp, scores, total_gts):
        if total_gts == 0 or len(scores) == 0:
            return np.array([0.0, 1.0]), np.array([1.0, 0.0])

        sort_idx = np.argsort(-scores)
        tp = tp[sort_idx]
        fp = fp[sort_idx]
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        recalls = np.concatenate(([0.0], cum_tp / total_gts))
        precisions = np.concatenate(([1.0], cum_tp / (cum_tp + cum_fp + 1e-10)))
        return recalls, precisions

    def plot_precision_recall_curve(self, results, save_path=None):
        """
        Plots and optionally saves the Precision-Recall curve.
        """
        pr_data = results.get("pr_curve", {})
        recalls = pr_data.get("recalls", [0, 1])
        precisions = pr_data.get("precisions", [1, 0])

        plt.figure(figsize=(7, 5))
        plt.plot(recalls, precisions, color="#1f77b4", lw=2, label=f"mAP50 = {results['mAP50']:.3f}")
        plt.fill_between(recalls, precisions, alpha=0.2, color="#1f77b4")
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title("Object Detection - Precision-Recall Curve (IoU=0.50)", fontsize=14, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="lower left", fontsize=11)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        return plt.gcf()
