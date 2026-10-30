import sys
from pathlib import Path

import numpy as np


# Ensure chatterbox package on path
PKG_ROOT = Path(__file__).resolve().parents[1] / "chatterbox" / "src"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from chatterbox.utils import trim_silence


def test_trim_silence_applies_fade_and_concatenates():
    sr = 1000
    wav = np.zeros(sr, dtype=np.float32)
    wav[200:400] = 1.0
    speech = [{"start": 0.2, "end": 0.4}]

    trimmed = trim_silence(wav, speech, sr, fade_duration=0.02)

    assert len(trimmed) == 200
    assert trimmed[0] == 0.0
    assert trimmed[20] == 1.0
    assert trimmed[-1] == 0.0
    assert trimmed[-21] == 1.0


def test_trim_silence_no_timestamps_returns_original():
    wav = np.arange(10, dtype=np.float32)
    out = trim_silence(wav, [], 1)
    assert np.array_equal(out, wav)
