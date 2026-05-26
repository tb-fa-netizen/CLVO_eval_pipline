"""
models/base.py
~~~~~~~~~~~~~~
Abstract base class that every TTS model wrapper must implement.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BaseTTSModel(ABC):
    """
    Unified interface for all TTS/voice-cloning backends.

    Subclasses must implement:
        load_model()     – load weights onto the configured device
        generate()       – produce a cloned waveform
        supports_language() – declare which language codes are supported
    """

    # Subclasses should override this with a human-readable name used in
    # output paths and metadata.
    MODEL_NAME: str = "BaseTTSModel"

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        languages: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        model_path : str
            HuggingFace model ID or local absolute path to model weights.
        device : str
            PyTorch device string, e.g. "cuda:0", "cuda:2", "cpu".
        languages : list[str] | None
            Language codes this model supports.  Falls back to the subclass
            default if None.
        **kwargs :
            Extra model-specific parameters (passed through from config).
        """
        self.model_path = model_path
        self.device = device
        self._supported_languages: List[str] = languages or []
        self.kwargs = kwargs
        self._model = None          # populated by load_model()
        self._is_loaded: bool = False

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights into memory on self.device."""
        ...

    @abstractmethod
    def generate(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str,
        language: str,
    ) -> np.ndarray:
        """
        Synthesise *text* in *language* with the voice from *ref_audio_path*.

        Parameters
        ----------
        text : str
            Target text to synthesise.
        ref_audio_path : str
            Path to the reference (prompt) audio file.
        ref_text : str
            Transcript of the reference audio (some models require this).
        language : str
            BCP-47 / ISO-639-1 language code, e.g. "en", "zh", "fr".

        Returns
        -------
        np.ndarray
            1-D float32 waveform (mono, normalised to [-1, 1]).
        """
        ...

    def supports_language(self, lang: str) -> bool:
        """Return True if this model can synthesise *lang*."""
        return lang in self._supported_languages

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def ensure_loaded(self) -> None:
        """Call load_model() exactly once."""
        if not self._is_loaded:
            logger.info("[%s] Loading model from %s onto %s …",
                        self.MODEL_NAME, self.model_path, self.device)
            self.load_model()
            self._is_loaded = True
            logger.info("[%s] Model ready.", self.MODEL_NAME)

    @property
    def name(self) -> str:
        return self.MODEL_NAME

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_path={self.model_path!r}, "
            f"device={self.device!r}, "
            f"languages={self._supported_languages})"
        )
