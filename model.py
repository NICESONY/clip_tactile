import torch
import torch.nn as nn
from transformers import CLIPVisionModel


class CLIPVisionRegressor(nn.Module):
    """
    CLIP vision encoder + MLP regression head

    freeze_strategy:
        "all"     — encoder 전체 freeze (head만 학습)
        "partial" — 마지막 unfreeze_layers개 encoder layer + post_layernorm unfreeze
        "none"    — encoder 전체 fine-tuning
    """

    def __init__(
        self,
        pretrained_model_name="openai/clip-vit-base-patch32",
        output_dim=6,
        hidden_dim=256,
        dropout=0.1,
        freeze_strategy="all",
        unfreeze_layers=2,
    ):
        super().__init__()

        self.vision_encoder = CLIPVisionModel.from_pretrained(pretrained_model_name)
        vision_hidden_dim = self.vision_encoder.config.hidden_size

        #  Freeze 전략 적용 
        if freeze_strategy == "all":
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        elif freeze_strategy == "partial":
            # 1) 전체 freeze
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

            # 2) 마지막 N개 encoder layer unfreeze
            encoder_layers = self.vision_encoder.vision_model.encoder.layers
            total_layers = len(encoder_layers)
            for i in range(total_layers - unfreeze_layers, total_layers):
                for param in encoder_layers[i].parameters():
                    param.requires_grad = True

            # 3) post_layernorm unfreeze
            for param in self.vision_encoder.vision_model.post_layernorm.parameters():
                param.requires_grad = True

        elif freeze_strategy == "none":
            pass  # 전체 fine-tuning

        else:
            raise ValueError(
                f"Unknown freeze_strategy: {freeze_strategy}. "
                f"Choose from 'all', 'partial', 'none'."
            )

        # 학습 파라미터 수 출력
        trainable = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.vision_encoder.parameters())
        print(f"[Encoder] freeze_strategy={freeze_strategy}  "
              f"trainable={trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

        self.head = nn.Sequential(
            nn.Linear(vision_hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # x: [B, C, H, W] = pixel_values
        outputs = self.vision_encoder(pixel_values=x)
        feat = outputs.pooler_output
        out = self.head(feat)
        return out
