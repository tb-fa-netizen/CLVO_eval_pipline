"""
models/moss_tts_model.py
~~~~~~~~~~~~~~~~~~~~~~~~
Wrapper around MOSS-TTS (OpenMOSS-Team) for voice cloning / continuation.

Because MOSS-TTS uses a multi-GPU layout via device_map="sequential" and
its own audio tokenizer, this wrapper closely mirrors the working script
from the user's codebase while fitting the BaseTTSModel contract.
"""

from __future__ import annotations

import importlib.util
import logging
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .base import BaseTTSModel

logger = logging.getLogger(__name__)


def _resolve_attn_implementation(device: str, dtype: torch.dtype) -> str:
    if (
        device.startswith("cuda")
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        dev_idx = int(device.split(":")[-1]) if ":" in device else 0
        major, _ = torch.cuda.get_device_capability(torch.device(dev_idx))
        if major >= 8:
            return "flash_attention_2"
    if device.startswith("cuda"):
        return "sdpa"
    return "eager"


class MossTTSModel(BaseTTSModel):
    """
    MOSS-TTS wrapper.

    This model uses a *continuation* approach: it is primed with a short
    reference audio + matching transcript, then asked to continue speaking
    new text in the same voice.
    """

    MODEL_NAME = "MossTTS"

    _DEFAULT_LANGUAGES: List[str] = ["en", "zh"]

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        languages: Optional[List[str]] = None,
        audio_tokenizer_device: Optional[str] = None,
        dtype: str = "bfloat16",
        attn_implementation: Optional[str] = None,
        max_memory: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            device=device,
            languages=languages or self._DEFAULT_LANGUAGES,
            **kwargs,
        )
        self._audio_tokenizer_device = audio_tokenizer_device or device
        self._dtype_str = dtype
        self._attn_impl_override = attn_implementation
        self._max_memory_raw: Dict[str, str] = max_memory or {}
        self._sample_rate: int = 24_000

    # ------------------------------------------------------------------ #
    # BaseTTSModel interface                                               #
    # ------------------------------------------------------------------ #

    def load_model(self) -> None:
        from transformers import AutoModel, AutoProcessor  # type: ignore

        # Disable broken cuDNN SDPA; keep flash + mem-efficient as fallback
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }
        dtype = dtype_map.get(self._dtype_str, torch.bfloat16)

        attn_impl = self._attn_impl_override or _resolve_attn_implementation(
            self.device, dtype
        )
        logger.info("[MossTTS] attn_implementation=%s, dtype=%s", attn_impl, dtype)

        # Build max_memory dict with integer keys (as required by HF)
        max_memory: Dict[Any, str] = {}
        for k, v in self._max_memory_raw.items():
            try:
                max_memory[int(k)] = v
            except ValueError:
                max_memory[k] = v

        self._processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        # Audio tokenizer must stay float32 (waveforms are float32)
        self._processor.audio_tokenizer = (
            self._processor.audio_tokenizer.to(self._audio_tokenizer_device)
        )

        model_kwargs: Dict[str, Any] = dict(
            trust_remote_code=True,
            attn_implementation=attn_impl,
            dtype=dtype,
        )
        if max_memory:
            model_kwargs["device_map"] = "sequential"
            model_kwargs["max_memory"] = max_memory
        else:
            model_kwargs["device_map"] = self.device

        self._model = AutoModel.from_pretrained(self.model_path, **model_kwargs)
        self._model.eval()

        # Determine primary device (where embed_tokens lives)
        self._primary_device = next(self._model.parameters()).device

        # Patch get_input_embeddings to avoid cross-device tensor errors when
        # model is spread across multiple GPUs by accelerate.
        _primary = self._primary_device

        def _patched_get_input_embeddings(self_inner, input_ids: torch.Tensor):
            inputs_embeds = self_inner.language_model.embed_tokens(
                input_ids[..., 0]
            )
            target_device = inputs_embeds.device
            for i, embed_layer in enumerate(self_inner.emb_ext):
                inputs_embeds = inputs_embeds + embed_layer(
                    input_ids[..., i + 1]
                ).to(target_device)
            return inputs_embeds

        self._model.get_input_embeddings = types.MethodType(
            _patched_get_input_embeddings, self._model
        )

        if hasattr(self._processor, "model_config"):
            self._sample_rate = self._processor.model_config.sampling_rate
        logger.info(
            "[MossTTS] Loaded.  Primary device: %s, sample_rate: %d Hz.",
            self._primary_device,
            self._sample_rate,
        )

    def generate(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str,
        language: str,
    ) -> np.ndarray:
        """
        Voice-clone *text* by priming the model with the reference audio
        using the *continuation* modality.
        """
        processor = self._processor
        model = self._model
        device = self._primary_device

        conversation = [
            processor.build_user_message(
                text=ref_text + text, reference=[ref_audio_path]
            ),
            processor.build_assistant_message(
                audio_codes_list=[ref_audio_path]
            ),
        ]

        batch = processor([conversation], mode="continuation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=4096,
                audio_temperature=1.0,
                audio_top_p=0.8,
                audio_top_k=25,
                audio_repetition_penalty=1.0,
            )

        messages = processor.decode(outputs)
        audio_codes = messages[0].audio_codes_list[0]
        # audio_codes is a 1-D or 2-D tensor; squeeze to 1-D float32
        wav = audio_codes.squeeze().cpu().numpy().astype(np.float32)
        return wav

    @property
    def sample_rate(self) -> int:
        return self._sample_rate
