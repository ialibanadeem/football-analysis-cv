import os
import numpy as np
from metrics import FootballAnalysisEvaluator, RuntimeTracker

def generate_sample_data():
    """
    Generates synthetic sample dataset for demonstrating metrics calculation across all 9 components.
    """
    np.random.seed(42)

    # 1. Object Detection (50 frames)
    od_preds, od_gts = [], []
    for _ in range(50):
        gt_boxes = [{"bbox": [100, 100, 150, 200], "class": 0}, {"bbox": [300, 200, 350, 300], "class": 0}]
        pred_boxes = [
            {"bbox": [102, 98, 151, 202], "score": 0.92, "class": 0},
            {"bbox": [298, 203, 349, 297], "score": 0.88, "class": 0},
            {"bbox": [500, 400, 520, 450], "score": 0.35, "class": 0}
        ]
        od_gts.append(gt_boxes)
        od_preds.append(pred_boxes)

    # 2. Player Tracking (50 frames)
    pt_preds, pt_gts = [], []
    for f in range(50):
        p1_gt = [100 + f*2, 100, 150 + f*2, 200]
        p2_gt = [300, 200 + f, 350, 300 + f]
        gt_dict = {1: p1_gt, 2: p2_gt}
        track_id1 = 1 if f < 25 else 99
        pred_dict = {track_id1: [p1_gt[0]+1, p1_gt[1]-1, p1_gt[2]+1, p1_gt[3]], 2: p2_gt}
        pt_gts.append(gt_dict)
        pt_preds.append(pred_dict)

    # 3. Ball Tracking (50 frames)
    bt_preds, bt_gts = [], []
    for f in range(50):
        ball_gt = {"bbox": [400 + f*3, 300, 415 + f*3, 315]}
        ball_pred = {"bbox": [402 + f*3, 299, 416 + f*3, 314]} if f % 5 != 0 else None
        bt_gts.append(ball_gt)
        bt_preds.append(ball_pred)

    # 4. Jersey Classification (100 samples)
    y_true = np.random.choice([1, 2], size=100)
    y_pred = y_true.copy()
    y_pred[np.random.choice(100, 8, replace=False)] = 3 - y_pred[np.random.choice(100, 8, replace=False)]

    # 5. Possession (100 frames)
    gt_possession = np.array([1]*40 + [2]*40 + [0]*20)
    pred_possession = gt_possession.copy()
    pred_possession[np.random.choice(100, 6, replace=False)] = 1

    # 6. Camera Compensation (50 frames)
    cam_movement = [[np.random.normal(5, 2), np.random.normal(-2, 1)] for _ in range(50)]

    # 7. Perspective Mapping (20 ground control points)
    gt_pts = np.array([[10, 20], [15, 35], [20, 50], [5, 60], [22, 10]])
    pred_pts = gt_pts + np.random.normal(0, 0.4, size=gt_pts.shape)

    # 8. Speed Estimation (30 speed evaluations)
    gt_speeds = np.linspace(10, 25, 30)
    pred_speeds = gt_speeds + np.random.normal(0, 0.8, size=30)

    # 9. Runtime Tracking
    tracker = RuntimeTracker()
    for _ in range(30):
        tracker.measure_frame(lambda: time_sleep_dummy())

    return {
        "object_detection": {"preds": od_preds, "gts": od_gts},
        "player_tracking": {"preds": pt_preds, "gts": pt_gts},
        "ball_tracking": {"preds": bt_preds, "gts": bt_gts},
        "jersey_classification": {"y_true": y_true, "y_pred": y_pred},
        "possession": {"pred_possession": pred_possession, "gt_possession": gt_possession},
        "camera_compensation": {"camera_movement_per_frame": cam_movement},
        "perspective_mapping": {"pred_pts": pred_pts, "gt_pts": gt_pts},
        "speed_estimation": {"pred_speeds": pred_speeds, "gt_speeds": gt_speeds},
        "runtime": {"tracker": tracker}
    }

def time_sleep_dummy():
    import time
    time.sleep(0.015)

def main():
    print("Running Football Analysis Metrics Evaluation & Saving Visual Results...")
    sample_data = generate_sample_data()

    evaluator = FootballAnalysisEvaluator()
    evaluator.evaluate_all(sample_data)

    # Print markdown table report
    report = evaluator.generate_markdown_report()
    print("\n" + report)

    # Save visual results as JPG files
    output_dir = "output_metrics"
    saved_files = evaluator.save_all_visual_results(output_dir=output_dir)

    print(f"\nVisual results successfully saved to '{output_dir}/' folder as JPG files:")
    for f in saved_files:
        print(f" - {os.path.basename(f)}")

if __name__ == "__main__":
    main()
