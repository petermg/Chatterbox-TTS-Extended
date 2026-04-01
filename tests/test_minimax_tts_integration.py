"""
Integration tests for MiniMax Cloud TTS.

These tests hit the real MiniMax API and are skipped when MINIMAX_API_KEY
is not set.  Run with:

    MINIMAX_API_KEY=<your-key> python -m pytest tests/test_minimax_tts_integration.py -v
"""

import os
import tempfile

import pytest

from minimax_tts import MiniMaxTTS, MiniMaxTTSError, MINIMAX_VOICES

_HAS_KEY = bool(os.environ.get("MINIMAX_API_KEY"))
skipif_no_key = pytest.mark.skipif(not _HAS_KEY, reason="MINIMAX_API_KEY not set")


@skipif_no_key
class TestMiniMaxTTSIntegration:

    def test_synthesize_short_text(self):
        tts = MiniMaxTTS()
        audio_bytes = tts.synthesize("Hello, this is a test.")
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 1000  # sanity: at least ~1 KB

    def test_synthesize_with_different_voice(self):
        tts = MiniMaxTTS(voice_id="Deep_Voice_Man")
        audio_bytes = tts.synthesize("Testing with a deep voice.")
        assert len(audio_bytes) > 1000

    def test_synthesize_to_wav_file(self):
        tts = MiniMaxTTS()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            result = tts.synthesize_to_file("Integration test wav output.", path)
            assert os.path.exists(result)
            assert os.path.getsize(result) > 1000

            import soundfile as sf
            data, sr = sf.read(result)
            assert sr > 0
            assert len(data) > 0
        finally:
            os.unlink(path)

    def test_synthesize_to_tensor(self):
        tts = MiniMaxTTS()
        tensor, sr = tts.synthesize_to_tensor("Tensor output test.")
        assert tensor.dim() == 2
        assert tensor.shape[0] == 1
        assert tensor.shape[1] > 0
        assert sr > 0

    def test_turbo_model(self):
        tts = MiniMaxTTS(model="speech-2.8-turbo")
        audio_bytes = tts.synthesize("Testing turbo model speed.")
        assert len(audio_bytes) > 500
