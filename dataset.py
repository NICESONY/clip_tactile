import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class TactileCoordinateDataset(Dataset):
    def __init__(self, csv_path, image_dir, output_dim=6, image_processor=None,
                 image_mean=None, image_std=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.output_dim = output_dim
        self.image_processor = image_processor
        self.image_mean = image_mean
        self.image_std = image_std

        if image_processor is None:
            raise ValueError("image_processor must be provided for CLIP model.")

        required_cols = ["image_name"]
        if output_dim == 2:
            required_cols += ["x", "y"]
        elif output_dim == 3:
            required_cols += ["x", "y", "z"]
        elif output_dim == 6:
            required_cols += ["dX", "dY", "dZ", "Fx", "Fy", "Fz"]
        else:
            raise ValueError(f"Unsupported output_dim: {output_dim}")

        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row["image_name"])
        image = Image.open(image_path).convert("RGB")

        if self.output_dim == 2:
            target = torch.tensor([row["x"], row["y"]], dtype=torch.float32)
        elif self.output_dim == 3:
            target = torch.tensor([row["x"], row["y"], row["z"]], dtype=torch.float32)
        elif self.output_dim == 6:
            target = torch.tensor(
                [row["dX"], row["dY"], row["dZ"], row["Fx"], row["Fy"], row["Fz"]],
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"Unsupported output_dim: {self.output_dim}")

        proc_kwargs = {"images": image, "return_tensors": "pt"}
        if self.image_mean is not None:
            # 패딩 모드: 이미 224x224이므로 resize/crop 불필요
            proc_kwargs["image_mean"] = self.image_mean
            proc_kwargs["image_std"] = self.image_std
            proc_kwargs["do_center_crop"] = False
            proc_kwargs["do_resize"] = False
        else:
            # 기존 모드: CLIP resize+crop + 프로젝트 커스텀 mean/std
            proc_kwargs["image_mean"] = [0.41613302234013877, 0.34324077486038207, 0.3261217144838969]
            proc_kwargs["image_std"] = [0.3449644943991341, 0.3238820460207181, 0.32351404629653335]

        processed = self.image_processor(**proc_kwargs)
        image_tensor = processed["pixel_values"].squeeze(0)

        return {
            "image": image_tensor,
            "target": target,
            "image_name": row["image_name"],
        }
