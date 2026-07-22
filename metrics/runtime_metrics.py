import time
import numpy as np
import matplotlib.pyplot as plt
import psutil

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class RuntimeTracker:
    """
    Measures FPS, Latency (ms), CPU %, RAM, and GPU memory usage.
    Can be used as a context manager per frame or across a batch.
    """
    def __init__(self):
        self.frame_latencies_ms = []
        self.cpu_usages = []
        self.ram_usages_mb = []
        self.gpu_memory_mb = []

    def measure_frame(self, func, *args, **kwargs):
        """
        Executes func(*args, **kwargs) and records latency & system resource stats.
        """
        start_time = time.perf_counter()
        
        # System status before
        cpu_pct = psutil.cpu_percent(interval=None)
        ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)

        result = func(*args, **kwargs)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0

        self.frame_latencies_ms.append(latency_ms)
        self.cpu_usages.append(cpu_pct)
        self.ram_usages_mb.append(ram_mb)

        if HAS_TORCH and torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)
            self.gpu_memory_mb.append(gpu_mem)
        else:
            self.gpu_memory_mb.append(0.0)

        return result

    def get_summary(self):
        latencies = np.array(self.frame_latencies_ms)
        if len(latencies) == 0:
            return {
                "FPS": 0.0,
                "latency_mean_ms": 0.0,
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "latency_p99_ms": 0.0,
                "CPU_usage_pct": 0.0,
                "RAM_usage_mb": 0.0,
                "GPU_memory_mb": 0.0,
                "GPU_device": "CPU / Unavailable"
            }

        total_time_sec = np.sum(latencies) / 1000.0
        fps = len(latencies) / total_time_sec if total_time_sec > 0 else 0.0

        gpu_device_name = "CPU / Unavailable"
        if HAS_TORCH and torch.cuda.is_available():
            gpu_device_name = torch.cuda.get_device_name(0)

        return {
            "FPS": float(fps),
            "latency_mean_ms": float(np.mean(latencies)),
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "latency_p99_ms": float(np.percentile(latencies, 99)),
            "CPU_usage_pct": float(np.mean(self.cpu_usages)) if self.cpu_usages else 0.0,
            "RAM_usage_mb": float(np.mean(self.ram_usages_mb)) if self.ram_usages_mb else 0.0,
            "GPU_memory_mb": float(np.mean(self.gpu_memory_mb)) if self.gpu_memory_mb else 0.0,
            "GPU_device": gpu_device_name,
            "raw_latencies": self.frame_latencies_ms
        }

    def plot_runtime_summary(self, summary=None, save_path=None):
        """
        Plots latency distribution histogram and resource usage summary.
        """
        if summary is None:
            summary = self.get_summary()

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

        latencies = summary.get("raw_latencies", [summary.get("latency_mean_ms", 30)])
        axes[0].hist(latencies, bins=20, color="#2ca02c", edgecolor="black", alpha=0.75)
        axes[0].axvline(summary["latency_mean_ms"], color="red", linestyle="dashed", linewidth=2, label=f"Mean: {summary['latency_mean_ms']:.1f} ms")
        axes[0].axvline(summary["latency_p95_ms"], color="orange", linestyle="dashed", linewidth=2, label=f"P95: {summary['latency_p95_ms']:.1f} ms")
        axes[0].set_xlabel("Inference Latency (ms)", fontsize=11)
        axes[0].set_ylabel("Frequency", fontsize=11)
        axes[0].set_title(f"Latency Distribution (FPS: {summary['FPS']:.1f})", fontsize=13, fontweight="bold")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Resource Usage Summary
        resources = ["CPU (%)", "RAM (100MB)", "GPU VRAM (100MB)"]
        values = [
            summary["CPU_usage_pct"],
            summary["RAM_usage_mb"] / 100.0,
            summary["GPU_memory_mb"] / 100.0
        ]
        colors = ["#1f77b4", "#ff7f0e", "#9467bd"]

        axes[1].bar(resources, values, color=colors, alpha=0.85)
        for i, (res, val) in enumerate(zip(resources, values)):
            actual_str = f"{summary['CPU_usage_pct']:.1f}%" if i == 0 else f"{val*100:.0f} MB"
            axes[1].text(i, val + 0.5, actual_str, ha="center", fontweight="bold")

        axes[1].set_title(f"System Resource Utilization ({summary['GPU_device']})", fontsize=13, fontweight="bold")
        axes[1].grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        return fig
