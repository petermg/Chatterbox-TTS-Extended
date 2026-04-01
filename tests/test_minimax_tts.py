"""
Unit tests for minimax_tts.py — MiniMax Cloud TTS provider.
"""

import io
import json
import os
import struct
import tempfile
from unittest import mock

import numpy as np
import pytest

from minimax_tts import (
    DEFAULT_MODEL,
    DEFAULT_VOICE,
    MINIMAX_TTS_ENDPOINT,
    MINIMAX_TTS_MODELS,
    MINIMAX_VOICES,
    MiniMaxTTS,
    MiniMaxTTSError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hex_mp3_stub(duration_ms: int = 200, sr: int = 24000) -> str:
    """Return hex-encoded MP3-ish bytes (actually a valid WAV so soundfile
    can decode it) for use as the ``data.audio`` API response field."""
    import soundfile as sf

    num_samples = int(sr * duration_ms / 1000)
    samples = np.zeros(num_samples, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    return buf.getvalue().hex()


def _api_success_response(hex_audio: str) -> dict:
    return {
        "base_resp": {"status_code": 0, "status_msg": "success"},
        "data": {"audio": hex_audio},
    }


def _api_error_response(code: int = 1000, msg: str = "bad request") -> dict:
    return {
        "base_resp": {"status_code": code, "status_msg": msg},
        "data": {},
    }


# ---------------------------------------------------------------------------
# Tests — construction
# ---------------------------------------------------------------------------


class TestMiniMaxTTSInit:
    def test_explicit_api_key(self):
        tts = MiniMaxTTS(api_key="test-key-123")
        assert tts.api_key == "test-key-123"
        assert tts.model == DEFAULT_MODEL
        assert tts.voice_id == DEFAULT_VOICE

    def test_env_var_api_key(self):
        with mock.patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key-456"}):
            tts = MiniMaxTTS()
            assert tts.api_key == "env-key-456"

    def test_missing_api_key_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MINIMAX_API_KEY", None)
            with pytest.raises(MiniMaxTTSError, match="API key is required"):
                MiniMaxTTS()

    def test_custom_model_and_voice(self):
        tts = MiniMaxTTS(
            api_key="k", model="speech-2.8-turbo", voice_id="Deep_Voice_Man"
        )
        assert tts.model == "speech-2.8-turbo"
        assert tts.voice_id == "Deep_Voice_Man"


# ---------------------------------------------------------------------------
# Tests — synthesize
# ---------------------------------------------------------------------------


class TestSynthesize:
    def _mock_post(self, hex_audio):
        resp = mock.Mock()
        resp.status_code = 200
        resp.raise_for_status = mock.Mock()
        resp.json.return_value = _api_success_response(hex_audio)
        return resp

    def test_synthesize_returns_bytes(self):
        hex_audio = _make_hex_mp3_stub()
        tts = MiniMaxTTS(api_key="key")
        with mock.patch("minimax_tts.requests.post", return_value=self._mock_post(hex_audio)) as m:
            result = tts.synthesize("Hello world")
            assert isinstance(result, bytes)
            assert len(result) > 0
            m.assert_called_once()
            call_kwargs = m.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert payload["text"] == "Hello world"
            assert payload["model"] == DEFAULT_MODEL
            assert payload["voice_setting"]["voice_id"] == DEFAULT_VOICE

    def test_synthesize_voice_override(self):
        hex_audio = _make_hex_mp3_stub()
        tts = MiniMaxTTS(api_key="key")
        with mock.patch("minimax_tts.requests.post", return_value=self._mock_post(hex_audio)) as m:
            tts.synthesize("Test", voice_id="Deep_Voice_Man")
            payload = m.call_args.kwargs.get("json") or m.call_args[1].get("json")
            assert payload["voice_setting"]["voice_id"] == "Deep_Voice_Man"

    def test_synthesize_model_override(self):
        hex_audio = _make_hex_mp3_stub()
        tts = MiniMaxTTS(api_key="key")
        with mock.patch("minimax_tts.requests.post", return_value=self._mock_post(hex_audio)) as m:
            tts.synthesize("Test", model="speech-2.8-turbo")
            payload = m.call_args.kwargs.get("json") or m.call_args[1].get("json")
            assert payload["model"] == "speech-2.8-turbo"

    def test_synthesize_speed_clamping(self):
        hex_audio = _make_hex_mp3_stub()
        tts = MiniMaxTTS(api_key="key")
        with mock.patch("minimax_tts.requests.post", return_value=self._mock_post(hex_audio)) as m:
            tts.synthesize("Test", speed=0.1)
            payload = m.call_args.kwargs.get("json") or m.call_args[1].get("json")
            assert payload["voice_setting"]["speed"] == 0.5

            tts.synthesize("Test", speed=5.0)
            payload = m.call_args.kwargs.get("json") or m.call_args[1].get("json")
            assert payload["voice_setting"]["speed"] == 2.0

    def test_synthesize_api_error(self):
        resp = mock.Mock()
        resp.status_code = 200
        resp.raise_for_status = mock.Mock()
        resp.json.return_value = _api_error_response(1000, "invalid key")
        tts = MiniMaxTTS(api_key="key")
        with mock.patch("minimax_tts.requests.post", return_value=resp):
            with pytest.raises(MiniMaxTTSError, match="invalid key"):
                tts.synthesize("Test")

    def test_synthesize_empty_audio(self):
        resp = mock.Mock()
        resp.status_code = 200
        resp.raise_for_status = mock.Mock()
        resp.json.return_value = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "data": {"audio": ""},
        }
        tts = MiniMaxTTS(api_key="key")
        with mock.patch("minimax_tts.requests.post", return_value=resp):
            with pytest.raises(MiniMaxTTSError, match="empty audio"):
                tts.synthesize("Test")

    def test_synthesize_http_error(self):
        resp = mock.Mock()
        resp.status_code = 500
        resp.raise_for_status.side_effect = Exception("Internal Server Error")
        tts = MiniMaxTTS(api_key="key")
        with mock.patch("minimax_tts.requests.post", return_value=resp):
            with pytest.raises(Exception, match="Internal Server Error"):
                tts.synthesize("Test")

    def test_synthesize_request_headers(self):
        hex_audio = _make_hex_mp3_stub()
        tts = MiniMaxTTS(api_key="my-secret-key")
        with mock.patch("minimax_tts.requests.post", return_value=self._mock_post(hex_audio)) as m:
            tts.synthesize("Hello")
            call_kwargs = m.call_args
            headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
            assert headers["Authorization"] == "Bearer my-secret-key"
            assert headers["Content-Type"] == "application/json"

    def test_synthesize_endpoint_url(self):
        hex_audio = _make_hex_mp3_stub()
        tts = MiniMaxTTS(api_key="key")
        with mock.patch("minimax_tts.requests.post", return_value=self._mock_post(hex_audio)) as m:
            tts.synthesize("Hello")
            assert m.call_args[0][0] == MINIMAX_TTS_ENDPOINT


# ---------------------------------------------------------------------------
# Tests — synthesize_to_file
# ---------------------------------------------------------------------------


class TestSynthesizeToFile:
    def test_wav_output(self):
        hex_audio = _make_hex_mp3_stub()
        tts = MiniMaxTTS(api_key="key")
        resp = mock.Mock()
        resp.status_code = 200
        resp.raise_for_status = mock.Mock()
        resp.json.return_value = _api_success_response(hex_audio)
        with mock.patch("minimax_tts.requests.post", return_value=resp):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                out = tts.synthesize_to_file("Hello", f.name)
                assert os.path.exists(out)
                assert os.path.getsize(out) > 0
                os.unlink(out)

    def test_mp3_output(self):
        hex_audio = _make_hex_mp3_stub()
        tts = MiniMaxTTS(api_key="key")
        resp = mock.Mock()
        resp.status_code = 200
        resp.raise_for_status = mock.Mock()
        resp.json.return_value = _api_success_response(hex_audio)
        with mock.patch("minimax_tts.requests.post", return_value=resp):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                out = tts.synthesize_to_file("Hello", f.name)
                assert os.path.exists(out)
                assert os.path.getsize(out) > 0
                os.unlink(out)


# ---------------------------------------------------------------------------
# Tests — synthesize_to_tensor
# ---------------------------------------------------------------------------


class TestSynthesizeToTensor:
    def test_returns_tensor_and_sr(self):
        hex_audio = _make_hex_mp3_stub(duration_ms=100, sr=24000)
        tts = MiniMaxTTS(api_key="key")
        resp = mock.Mock()
        resp.status_code = 200
        resp.raise_for_status = mock.Mock()
        resp.json.return_value = _api_success_response(hex_audio)
        with mock.patch("minimax_tts.requests.post", return_value=resp):
            tensor, sr = tts.synthesize_to_tensor("Hello")
            assert tensor.dim() == 2
            assert tensor.shape[0] == 1
            assert tensor.shape[1] > 0
            assert sr > 0

    def test_tensor_dtype_is_float(self):
        hex_audio = _make_hex_mp3_stub()
        tts = MiniMaxTTS(api_key="key")
        resp = mock.Mock()
        resp.status_code = 200
        resp.raise_for_status = mock.Mock()
        resp.json.return_value = _api_success_response(hex_audio)
        with mock.patch("minimax_tts.requests.post", return_value=resp):
            tensor, sr = tts.synthesize_to_tensor("Hello")
            import torch
            assert tensor.dtype == torch.float32


# ---------------------------------------------------------------------------
# Tests — constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_models_non_empty(self):
        assert len(MINIMAX_TTS_MODELS) >= 2

    def test_voices_non_empty(self):
        assert len(MINIMAX_VOICES) >= 10

    def test_default_model_in_models(self):
        assert DEFAULT_MODEL in MINIMAX_TTS_MODELS

    def test_default_voice_in_voices(self):
        assert DEFAULT_VOICE in MINIMAX_VOICES

    def test_endpoint_is_https(self):
        assert MINIMAX_TTS_ENDPOINT.startswith("https://")


# ---------------------------------------------------------------------------
# Tests — Chatter.py helper functions
# ---------------------------------------------------------------------------


class TestChatterHelpers:
    """Test the _minimax_tts_models() and _minimax_tts_voices() functions."""

    def test_minimax_tts_models_returns_dict(self):
        import importlib
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        # We test the standalone functions directly
        from minimax_tts import MINIMAX_TTS_MODELS, MINIMAX_VOICES
        assert isinstance(MINIMAX_TTS_MODELS, dict)
        assert "speech-2.8-hd" in MINIMAX_TTS_MODELS
        assert "speech-2.8-turbo" in MINIMAX_TTS_MODELS

    def test_minimax_voices_has_expected_entries(self):
        from minimax_tts import MINIMAX_VOICES
        expected = [
            "English_Graceful_Lady", "Deep_Voice_Man", "Friendly_Person",
            "sweet_girl", "cute_boy",
        ]
        for v in expected:
            assert v in MINIMAX_VOICES


# ---------------------------------------------------------------------------
# Tests — audio format
# ---------------------------------------------------------------------------


class TestAudioFormat:
    def test_audio_setting_format_mp3(self):
        """Verify the API payload always requests mp3 format."""
        hex_audio = _make_hex_mp3_stub()
        tts = MiniMaxTTS(api_key="key")
        resp = mock.Mock()
        resp.status_code = 200
        resp.raise_for_status = mock.Mock()
        resp.json.return_value = _api_success_response(hex_audio)
        with mock.patch("minimax_tts.requests.post", return_value=resp) as m:
            tts.synthesize("test")
            payload = m.call_args.kwargs.get("json") or m.call_args[1].get("json")
            assert payload["audio_setting"]["format"] == "mp3"
            # Must NOT contain sample_rate or bitrate (rejected by MiniMax API)
            assert "sample_rate" not in payload["audio_setting"]
            assert "bitrate" not in payload["audio_setting"]
