import os
import csv
import json
import random
from datetime import datetime

import numpy as np
import torch


#  Seed 
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#  Metrics 
def mae_metric(pred, target):
    return torch.mean(torch.abs(pred - target)).item()


def rmse_metric(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def euclidean_distance_metric(pred, target):
    dist = torch.norm(pred - target, dim=1)
    return dist.mean().item()


#  Checkpoint 
def save_checkpoint(model, optimizer, epoch, best_val_loss, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        save_path,
    )


#  Training Logger 
class TrainLogger:
    """매 epoch 결과를 CSV로 저장. 파일 헤더에 시작시간/파라미터 기록."""

    def __init__(self, log_dir, experiment_name, config):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{experiment_name}_{timestamp}.csv")
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.config = config
        self._header_written = False
        self._file = None
        self._writer = None

        # 파일 맨 앞에 메타정보를 주석으로 기록
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"# start_time: {self.start_time}\n")
            f.write(f"# config: {json.dumps(config, ensure_ascii=False)}\n")

    def log(self, metrics: dict):
        """metrics dict를 한 줄로 기록. 첫 호출 시 컬럼 헤더 작성."""
        if not self._header_written:
            self._file = open(self.log_path, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=list(metrics.keys()))
            self._writer.writeheader()
            self._header_written = True

        self._writer.writerow(metrics)
        self._file.flush()

    def close(self):
        if self._file:
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._file.write(f"# end_time: {end_time}\n")
            self._file.close()
