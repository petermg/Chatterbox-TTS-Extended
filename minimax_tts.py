"""
MiniMax Cloud TTS Provider for Chatterbox-TTS-Extended.

Provides cloud-based text-to-speech via the MiniMax T2A v2 API as an
alternative to the local Chatterbox model.  Useful when GPU resources
are limited or when a different voice palette is desired.

API reference: https://platform.minimaxi.com/document/T2A%20V2
"""

import io
import os
import struct
import tempfile
from typing import Optional

import numpy as np
import requests
import soundfile as sf
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MINIMAX_TTS_ENDPOINT = "https://api.minimax.io/v1/t2a_v2"

MINIMAX_TTS_MODELS = {
    "speech-2.8-hd": "High-definition quality, slower",
    "speech-2.8-turbo": "Fast generation, slightly lower quality",
}

MINIMAX_VOICES = {
    "English_Graceful_Lady": "Graceful female voice",
    "English_Insightful_Speaker": "Insightful male voice",
    "English_radiant_girl": "Radiant young female voice",
    "English_Persuasive_Man": "Persuasive male voice",
    "English_Lucky_Robot": "Robotic voice",
    "Wise_Woman": "Wise female voice",
    "cute_boy": "Cute boy voice",
    "lovely_girl": "Lovely girl voice",
    "Friendly_Person": "Friendly neutral voice",
    "Inspirational_girl": "Inspirational female voice",
    "Deep_Voice_Man": "Deep male voice",
    "sweet_girl": "Sweet girl voice",
}

DEFAULT_VOICE = "Friendly_Person"
DEFAULT_MODEL = "speech-2.8-hd"


# ---------------------------------------------------------------------------
# MiniMax TTS Client
# ---------------------------------------------------------------------------


class MiniMaxTTSError(Exception):
    """Raised when the MiniMax TTS API returns an error."""


class MiniMaxTTS:
    """Thin wrapper around the MiniMax T2A v2 API.

    Parameters
    ----------
    api_key : str, optional
        MiniMax API key.  Falls back to the ``MINIMAX_API_KEY`` environment
        variable when not provided.
    model : str
        One of ``speech-2.8-hd`` or ``speech-2.8-turbo``.
    voice_id : str
        A valid MiniMax voice identifier (see ``MINIMAX_VOICES``).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        voice_id: str = DEFAULT_VOICE,
    ) -> None:
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        if not self.api_key:
            raise MiniMaxTTSError(
                "MiniMax API key is required.  Set the MINIMAX_API_KEY "
                "environment variable or pass it explicitly."
            )
        self.model = model
        self.voice_id = voice_id

    # ---- public API --------------------------------------------------------

    def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        model: Optional[str] = None,
        speed: float = 1.0,
    ) -> bytes:
        """Synthesize *text* and return raw **MP3** bytes.

        Parameters
        ----------
        text : str
            The text to synthesize.
        voice_id : str, optional
            Override the default voice for this call.
        model : str, optional
            Override the default model for this call.
        speed : float
            Speech speed multiplier (0.5 – 2.0).

        Returns
        -------
        bytes
            MP3-encoded audio data.
        """
        payload = {
            "model": model or self.model,
            "text": text,
            "voice_setting": {
                "voice_id": voice_id or self.voice_id,
                "speed": max(0.5, min(speed, 2.0)),
            },
            "audio_setting": {
                "format": "mp3",
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            MINIMAX_TTS_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=120,
        )
        resp.raise_for_status()
        body = resp.json()

        # API returns base_resp with status_code
        base_resp = body.get("base_resp", {})
        status_code = base_resp.get("status_code", 0)
        if status_code != 0:
            raise MiniMaxTTSError(
                f"MiniMax TTS API error {status_code}: "
                f"{base_resp.get('status_msg', 'unknown error')}"
            )

        hex_audio = body.get("data", {}).get("audio", "")
        if not hex_audio:
            raise MiniMaxTTSError("MiniMax TTS API returned empty audio data.")
        return bytes.fromhex(hex_audio)

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        voice_id: Optional[str] = None,
        model: Optional[str] = None,
        speed: float = 1.0,
    ) -> str:
        """Synthesize *text* and write the result to *output_path*.

        If *output_path* ends with ``.wav``, the MP3 is decoded and re-saved
        as WAV so it is compatible with the rest of the pipeline.

        Returns the final file path.
        """
        mp3_bytes = self.synthesize(
            text, voice_id=voice_id, model=model, speed=speed
        )

        if output_path.lower().endswith(".wav"):
            wav_data, sr = sf.read(io.BytesIO(mp3_bytes))
            sf.write(output_path, wav_data, sr)
        else:
            with open(output_path, "wb") as f:
                f.write(mp3_bytes)
        return output_path

    def synthesize_to_tensor(
        self,
        text: str,
        voice_id: Optional[str] = None,
        model: Optional[str] = None,
        speed: float = 1.0,
    ) -> tuple:
        """Synthesize and return ``(waveform_tensor, sample_rate)``.

        The tensor shape is ``(1, num_samples)`` to match the format used
        by the local Chatterbox pipeline.
        """
        mp3_bytes = self.synthesize(
            text, voice_id=voice_id, model=model, speed=speed
        )
        wav_data, sr = sf.read(io.BytesIO(mp3_bytes))
        if wav_data.ndim > 1:
            wav_data = wav_data.mean(axis=1)
        tensor = torch.from_numpy(wav_data).unsqueeze(0).float()
        return tensor, sr
