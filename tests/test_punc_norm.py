import sys
from pathlib import Path

# Ensure chatterbox package on path
PKG_ROOT = Path(__file__).resolve().parents[1] / "chatterbox" / "src"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from chatterbox.tts import punc_norm


def test_punc_norm_empty_string():
    assert punc_norm("") == "You need to add some text for me to talk."


def test_punc_norm_handles_uncommon_punctuation():
    text = "hello… \u201cworld\u201d \u2014 test"
    expected = 'Hello,  "world" - test.'
    assert punc_norm(text) == expected
