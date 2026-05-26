"""
models/cosyvoice_model.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Wrapper around CosyVoice3 (0.5B) for zero-shot cross-lingual voice cloning.

Expects:
    config.models.CosyVoice3.repo_path  – root of the cloned CosyVoice repo
    config.models.CosyVoice3.model_path – pretrained weights directory
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import BaseTTSModel

logger = logging.getLogger(__name__)


class CosyVoiceModel(BaseTTSModel):
    """CosyVoice3 0.5B zero-shot voice cloning wrapper."""

    MODEL_NAME = "CosyVoice3"

    # Language codes CosyVoice3 supports out of the box
    _DEFAULT_LANGUAGES: List[str] = [
        "en", "ar", "de", "fa", "fr", "ja", "nl", "pt", "ru", "tr", "zh",
    ]

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        languages: Optional[List[str]] = None,
        repo_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            device=device,
            languages=languages or self._DEFAULT_LANGUAGES,
            **kwargs,
        )
        self._repo_path = repo_path

    # ------------------------------------------------------------------ #
    # BaseTTSModel interface                                               #
    # ------------------------------------------------------------------ #

    def load_model(self) -> None:
        # Inject CosyVoice repo and its Matcha-TTS dependency into sys.path
        if self._repo_path:
            repo = Path(self._repo_path)
            for p in [repo, repo / "third_party" / "Matcha-TTS"]:
                s = str(p)
                if s not in sys.path:
                    sys.path.insert(0, s)

        from cosyvoice.cli.cosyvoice import CosyVoice3  # type: ignore
        from cosyvoice.utils.file_utils import load_wav  # type: ignore

        self._load_wav = load_wav
        self._cosyvoice = CosyVoice3(self.model_path)
        self._sample_rate: int = self._cosyvoice.sample_rate

    def generate(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str,
        language: str,
    ) -> np.ndarray:
        """Generate cross-lingual cloned speech via zero-shot inference."""
        import torch

        prompt_16k = self._load_wav(ref_audio_path, 16_000)

        chunks = []
        for chunk in self._cosyvoice.inference_zero_shot(
            text,
            ref_text,
            prompt_16k,
            stream=False,
        ):
            chunks.append(chunk["tts_speech"])

        audio_tensor = torch.concat(chunks, dim=1).squeeze(0)
        return audio_tensor.cpu().numpy().astype(np.float32)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate
