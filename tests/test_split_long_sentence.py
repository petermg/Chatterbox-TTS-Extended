from pathlib import Path
import ast


def _load_split_long_sentence():
    file_path = Path(__file__).resolve().parents[1] / "Chatter.py"
    source = file_path.read_text()
    module_ast = ast.parse(source)
    func_node = next(
        node for node in module_ast.body
        if isinstance(node, ast.FunctionDef) and node.name == "split_long_sentence"
    )
    module = ast.Module(body=[func_node], type_ignores=[])
    code = compile(module, filename=str(file_path), mode="exec")
    namespace = {}
    exec(code, namespace)
    return namespace["split_long_sentence"]


split_long_sentence = _load_split_long_sentence()


def test_split_long_sentence_empty_string():
    assert split_long_sentence("", max_len=10) == [""]


def test_split_long_sentence_no_separators():
    sentence = "a" * 50
    assert split_long_sentence(sentence, max_len=10) == ["a" * 10] * 5


def test_split_long_sentence_with_spaces():
    sentence = ("word " * 15).strip()
    expected = [
        "word word word word",
        "word word word word",
        "word word word word",
        "word word word",
    ]
    assert split_long_sentence(sentence, max_len=20) == expected
