import sys
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import soundfile as sf


class DummyVCModel:
    sr = 16000

    def generate(self, input_audio_path, target_voice_path, apply_watermark=True, pitch_shift=0):
        wav, _ = sf.read(input_audio_path)
        return torch.from_numpy(wav).unsqueeze(0)


def test_voice_conversion_cleans_up_temp_files(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import types
    chatterbox_mod = types.ModuleType("chatterbox")
    chatterbox_src = types.ModuleType("chatterbox.src")
    inner_cb = types.ModuleType("chatterbox.src.chatterbox")
    tts_mod = types.ModuleType("chatterbox.src.chatterbox.tts")
    vc_mod = types.ModuleType("chatterbox.src.chatterbox.vc")
    class ChatterboxTTS:
        pass
    class ChatterboxVC:
        pass
    tts_mod.ChatterboxTTS = ChatterboxTTS
    vc_mod.ChatterboxVC = ChatterboxVC
    chatterbox_mod.src = chatterbox_src
    chatterbox_src.chatterbox = inner_cb
    sys.modules.update({
        "chatterbox": chatterbox_mod,
        "chatterbox.src": chatterbox_src,
        "chatterbox.src.chatterbox": inner_cb,
        "chatterbox.src.chatterbox.tts": tts_mod,
        "chatterbox.src.chatterbox.vc": vc_mod,
    })
    import Chatter

    monkeypatch.setattr(Chatter, "get_or_load_vc_model", lambda: DummyVCModel())

    sr = 16000
    duration = 2  # seconds
    wav = np.zeros(sr * duration)
    input_path = tmp_path / "input.wav"
    target_path = tmp_path / "target.wav"
    sf.write(input_path, wav, sr)
    sf.write(target_path, wav, sr)

    temp_dir = tempfile.gettempdir()
    before = {d for d in os.listdir(temp_dir) if d.startswith("vc_chunk_")}

    Chatter.voice_conversion(str(input_path), str(target_path), chunk_sec=0.5)

    after = {d for d in os.listdir(temp_dir) if d.startswith("vc_chunk_")}
    assert before == after

    # Clean up imported chatterbox package so other tests can import it freshly
    for m in list(sys.modules.keys()):
        if m.startswith("chatterbox"):
            sys.modules.pop(m)
