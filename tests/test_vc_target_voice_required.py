import sys
from pathlib import Path

import numpy as np
import torch
import pytest

# Ensure chatterbox package is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chatterbox" / "src"))

from chatterbox.vc import ChatterboxVC


class DummyS3Gen:
    def tokenizer(self, audio):
        return torch.zeros((1, 1)), None

    def inference(self, speech_tokens, ref_dict):
        return torch.tensor([[0.1, 0.2, 0.3]]), None


def fake_load(path, sr):
    return np.zeros(sr), sr


def test_generate_requires_target_voice(monkeypatch):
    monkeypatch.setattr("chatterbox.vc.librosa.load", fake_load)
    vc = ChatterboxVC(s3gen=DummyS3Gen(), device="cpu", ref_dict=None)
    with pytest.raises(RuntimeError):
        vc.generate("dummy.wav")
