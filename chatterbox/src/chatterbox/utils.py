import numpy as np
from typing import Iterable, Dict, List


def trim_silence(wav: np.ndarray, speech_timestamps: Iterable[Dict[str, float]], sr: int, fade_duration: float = 0.01) -> np.ndarray:
    """Trim non-speech portions from an audio waveform.

    Parameters
    ----------
    wav : np.ndarray
        1-D array of audio samples.
    speech_timestamps : Iterable[dict]
        Iterable of mappings with ``start`` and ``end`` times (in seconds)
        describing regions of speech in ``wav``.
    sr : int
        Sample rate of ``wav``.
    fade_duration : float, optional
        Length of the linear fade (in seconds) applied to the start and end
        of each speech segment. Defaults to 10 ms.

    Returns
    -------
    np.ndarray
        A waveform containing the voiced portions of ``wav`` concatenated
        together with fades applied. If ``speech_timestamps`` is empty, the
        original ``wav`` is returned unchanged.
    """
    speech_timestamps = list(speech_timestamps)
    if len(speech_timestamps) == 0:
        return wav

    fade_len = int(fade_duration * sr)
    segs: List[np.ndarray] = []
    for segment in speech_timestamps:
        start = int(segment["start"] * sr)
        end = int(segment["end"] * sr)
        seg = wav[start:end].copy()
        if seg.size == 0:
            continue
        fl = min(fade_len, seg.size // 2)
        if fl > 0:
            fade_in = np.linspace(0.0, 1.0, fl, endpoint=False)
            fade_out = np.linspace(1.0, 0.0, fl, endpoint=True)
            seg[:fl] *= fade_in
            seg[-fl:] *= fade_out
        segs.append(seg)

    return np.concatenate(segs) if segs else wav
