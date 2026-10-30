import sys
from pathlib import Path

import numpy as np
import torch

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


class DummyWatermarker:
    def apply_watermark(self, wav, sample_rate):
        return wav + 1


def create_vc(monkeypatch, watermarker=None):
    monkeypatch.setattr("chatterbox.vc.librosa.load", fake_load)
    s3gen = DummyS3Gen()
    vc = ChatterboxVC(s3gen=s3gen, device="cpu", ref_dict={})
    vc.watermarker = watermarker
    return vc


def test_generate_applies_watermark_when_available(monkeypatch):
    vc = create_vc(monkeypatch, watermarker=DummyWatermarker())
    out = vc.generate("dummy.wav", apply_watermark=True)
    expected = torch.tensor([1.1, 1.2, 1.3]).unsqueeze(0)
    assert torch.allclose(out, expected)


def test_generate_skips_watermark_when_unavailable(monkeypatch):
    vc = create_vc(monkeypatch, watermarker=None)
    out = vc.generate("dummy.wav", apply_watermark=True)
    expected = torch.tensor([0.1, 0.2, 0.3]).unsqueeze(0)
    assert torch.allclose(out, expected)


def test_generate_skips_watermark_when_flag_false(monkeypatch):
    vc = create_vc(monkeypatch, watermarker=DummyWatermarker())
    out = vc.generate("dummy.wav", apply_watermark=False)
    expected = torch.tensor([0.1, 0.2, 0.3]).unsqueeze(0)
    assert torch.allclose(out, expected)
