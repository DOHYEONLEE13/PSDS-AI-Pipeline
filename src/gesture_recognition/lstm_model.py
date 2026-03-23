"""src/gesture_recognition/lstm_model.py — SOS 모션 분류 LSTM 모델."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn as nn


class MotionClass(IntEnum):
    """SOS 모션 분류 클래스 인덱스."""

    NORMAL = 0
    SOS = 1


@dataclass
class ModelConfig:
    """SOSMotionLSTM 하이퍼파라미터.

    Attributes:
        input_size: 입력 피처 수 (21관절 × 3좌표 = 63).
        hidden_size: LSTM 히든 유닛 수.
        num_layers: LSTM 레이어 수.
        num_classes: 출력 클래스 수 (NORMAL + SOS = 2).
        dropout: 드롭아웃 비율.
        seq_len: 입력 시퀀스 길이 (프레임 수).
    """

    input_size: int = 63
    hidden_size: int = 128
    num_layers: int = 2
    num_classes: int = 2
    dropout: float = 0.3
    seq_len: int = 30


class SOSMotionLSTM(nn.Module):
    """30프레임 × 21관절 × 3좌표 입력을 받아 모션 클래스를 분류하는 LSTM 모델.

    마지막 타임스텝의 히든 상태를 선형 분류기에 통과시켜 로짓을 반환합니다.

    Args:
        config: 모델 하이퍼파라미터. None이면 기본값(ModelConfig)을 사용합니다.

    Examples:
        >>> model = SOSMotionLSTM()
        >>> x = torch.zeros(4, 30, 63)  # (batch, seq_len, features)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([4, 2])
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        self.config = cfg

        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.hidden_size, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순방향 전파.

        Args:
            x: (batch, seq_len, input_size) 형태의 입력 텐서.

        Returns:
            (batch, num_classes) 형태의 로짓 텐서.
        """
        out, _ = self.lstm(x)           # (batch, seq_len, hidden_size)
        last_hidden = out[:, -1, :]     # 마지막 타임스텝만 사용
        dropped = self.dropout(last_hidden)
        return self.classifier(dropped)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """소프트맥스 확률을 반환합니다 (no_grad 적용).

        Args:
            x: (batch, seq_len, input_size) 형태의 입력 텐서.

        Returns:
            (batch, num_classes) 형태의 확률 텐서.
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            return torch.softmax(logits, dim=-1)
