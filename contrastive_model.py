import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class CLIPContrastive(nn.Module):
    """
    CLIP image encoder + text encoder for contrastive learning.
    Text input = 6-axis force/torque values converted to a string.
    """

    def __init__(
        self,
        pretrained_model_name="openai/clip-vit-base-patch32",
        freeze_image_encoder=False,
        freeze_text_encoder=False,
        learnable_temperature=True,
        init_temperature=0.07,
    ):
        super().__init__()

        self.clip = CLIPModel.from_pretrained(pretrained_model_name)

        if freeze_image_encoder:
            for param in self.clip.vision_model.parameters():
                param.requires_grad = False
            for param in self.clip.visual_projection.parameters():
                param.requires_grad = False

        if freeze_text_encoder:
            for param in self.clip.text_model.parameters():
                param.requires_grad = False
            for param in self.clip.text_projection.parameters():
                param.requires_grad = False

        # learnable temperature (log scale)
        if learnable_temperature:
            self.log_temperature = nn.Parameter(
                torch.tensor(init_temperature).log()
            )
        else:
            self.register_buffer(
                "log_temperature", torch.tensor(init_temperature).log()
            )

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def encode_image(self, pixel_values):
        """Returns L2-normalized image embeddings [B, D]."""
        vision_out = self.clip.vision_model(pixel_values=pixel_values)
        image_embeds = self.clip.visual_projection(vision_out.pooler_output)
        return F.normalize(image_embeds, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        """Returns L2-normalized text embeddings [B, D]."""
        text_out = self.clip.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        text_embeds = self.clip.text_projection(text_out.pooler_output)
        return F.normalize(text_embeds, dim=-1)

    def forward(self, pixel_values, input_ids, attention_mask):
        """
        Returns:
            logits_per_image: [B, B] similarity matrix (image -> text)
            logits_per_text:  [B, B] similarity matrix (text -> image)
        """
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)

        logits_per_image = (image_embeds @ text_embeds.t()) / self.temperature
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


def clip_contrastive_loss(logits_per_image, logits_per_text):
    """Symmetric cross-entropy loss (same as original CLIP)."""
    batch_size = logits_per_image.size(0)
    labels = torch.arange(batch_size, device=logits_per_image.device)

    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)

    return (loss_i2t + loss_t2i) / 2.0


def supervised_contrastive_loss(logits_per_image, logits_per_text, text_labels):
    """
    Supervised Contrastive Loss: 같은 라벨 텍스트를 가진 쌍을 모두 positive로 처리.

    Args:
        logits_per_image: [B, B] image->text similarity
        logits_per_text:  [B, B] text->image similarity
        text_labels: list of str, 길이 B — 각 샘플의 텍스트 라벨

    기존 CLIP loss는 대각선만 positive (1:1 매칭).
    이 loss는 같은 텍스트를 가진 모든 쌍을 positive로 처리하여
    grid 양자화처럼 중복 라벨이 많은 경우에 적합.
    """
    B = logits_per_image.size(0)
    device = logits_per_image.device

    # 같은 라벨이면 1, 다르면 0인 마스크 [B, B]
    match_matrix = torch.zeros(B, B, device=device)
    for i in range(B):
        for j in range(B):
            if text_labels[i] == text_labels[j]:
                match_matrix[i, j] = 1.0

    # soft target: 각 행에서 positive 개수로 나눠서 확률 분포로 만듦
    # image->text: 각 이미지에 대해 같은 라벨인 텍스트들이 정답
    target_i2t = match_matrix / match_matrix.sum(dim=1, keepdim=True)
    # text->image: 각 텍스트에 대해 같은 라벨인 이미지들이 정답
    target_t2i = match_matrix.t() / match_matrix.t().sum(dim=1, keepdim=True)

    loss_i2t = (-target_i2t * F.log_softmax(logits_per_image, dim=1)).sum(dim=1).mean()
    loss_t2i = (-target_t2i * F.log_softmax(logits_per_text, dim=1)).sum(dim=1).mean()

    return (loss_i2t + loss_t2i) / 2.0
