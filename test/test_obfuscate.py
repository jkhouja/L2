import unicodedata as ud

import numpy as np
import pandas as pd
import pytest

from MultiLingOly.creation.code import obfuscate
from MultiLingOly.test import utils as u


def test_extract_pattern():
    input = "@@@ hi @@@"
    extracted = obfuscate.extract_pattern(input, "@@@")
    assert extracted == ["@@@ hi @@@"]

    input = "Extra char at beginning @@@ hi @@@"
    extracted = obfuscate.extract_pattern(input, "@@@")
    assert extracted == ["@@@ hi @@@"]

    input = "@@@ hi @@@ Extra char at end"
    extracted = obfuscate.extract_pattern(input, "@@@")
    assert extracted == ["@@@ hi @@@"]

    input = "@@@@@@"
    extracted = obfuscate.extract_pattern(input, "@@@")
    assert extracted == ["@@@@@@"]

    input = "@@@ hi \n line 2 @@@"
    extracted = obfuscate.extract_pattern(input, "@@@")
    assert extracted == ["@@@ hi \n line 2 @@@"]

    input = "@@@ hi \n line 2 @@@\nline3"
    extracted = obfuscate.extract_pattern(input, "@@@")
    assert extracted == ["@@@ hi \n line 2 @@@"]

    input = "\n\nline0@@@ hi \n line 2 @@@"
    extracted = obfuscate.extract_pattern(input, "@@@")
    assert extracted == ["@@@ hi \n line 2 @@@"]


@pytest.mark.parametrize(
    "inp, expected, mapping, test_comment",
    u.testdata + u.swapping_data + u.letter_case_data,
)
def test_replace_with_dict(inp, expected, mapping, test_comment):
    mapping = {
        ud.normalize("NFKD", k): ud.normalize("NFKD", v) for k, v in mapping.items()
    }
    inp = ud.normalize("NFKD", inp)
    out = obfuscate.replace_with_dictionary(inp, mapping)


@pytest.mark.parametrize(
    "inp, expected, mapping, test_comment",
    u.testdata + u.swapping_data + u.letter_case_data,
)
def test_obfuscation_output(inp, expected, mapping, test_comment):
    mapping = {
        ud.normalize("NFKD", k): ud.normalize("NFKD", v) for k, v in mapping.items()
    }
    inp = obfuscate.OBF_PATTERN + inp + obfuscate.OBF_PATTERN
    out, _ = obfuscate.obfuscate_text(inp, mapping)
    assert out == expected, f"failed: {test_comment}"


def test_letter_casing():
    inp = "@@@Hi there@@@"
    expected = "Hello there"
    mapping = {"hi": "hello"}
    out, case_issues = obfuscate.obfuscate_text(inp, mapping, ignorecase=True)
    assert case_issues == 0
    assert out == expected

    inp = "@@@HI there@@@"
    expected = "hello there"
    out, case_issues = obfuscate.obfuscate_text(inp, mapping, ignorecase=True)
    assert case_issues == 1
    assert out == expected

    inp = "@@@Hi there@@@"
    expected = "Hi there"
    out, case_issues = obfuscate.obfuscate_text(inp, mapping, ignorecase=False)
    assert case_issues == 0
    assert out == expected


def escape_unicode():
    inp = "@@@A\\u0161 nen\\u00f3riu \\u017eem\\u0117lap\\u012f@@@"
    mapping = {
        "Danute": "Danute",
        "Jokubas": "Jokubas",
        "Regina": "Regina",
        "Matis": "Matis",
        "dv": "s\u030c",
        "g": "t",
        "j": "k",
        "k": "g",
        "l": "z\u030c",
        "m": "dv",
        "n": "s",
        "p": "n",
        "r": "p",
        "s": "j",
        "t": "l",
        "s\u030c": "r",
        "z\u030c": "m",
        "a": "y",
        "e": "i",
        "i": "u\u0304",
        "o": "i\u0308",
        "u": "e\u0307",
        "y": "a",
        "i\u0308": "i\u0328",
        "o\u0301": "o",
        "a\u0328": "e",
        "e\u0307": "o\u0301",
        "i\u0328": "a\u0328",
        "u\u0304": "u",
    }

    expected = "Yr sisopu\u0304e\u0307 midvo\u0301z\u030cyna\u0328"
    out, case_issues = obfuscate.obfuscate_text(
        inp, mapping, ignorecase=False, unicode_escape=True
    )
    assert out == expected


def test_create_dict():
    toy_df = pd.DataFrame(
        data={
            "Num": [0],
            "Fixed": "test",
            "Set1": "a,b,c",
            "Table1": np.nan,
            "FreeTable1": "{d:e,f,g},{h:i,j,k}",
        }
    )
    obf_dict, fixed = obfuscate.create_obf_dictionary(toy_df, q_num=0)
    expected = {
        "FreeTable1": [[["d"], ["e", "f", "g"]], [["h"], ["i", "j", "k"]]],
        "Set1": [[["a", "b", "c"]]],
    }
    expected_fixed = {"Test": [[["Test"]]], "test": [[["test"]]]}
    assert obf_dict == expected, obf_dict
    assert fixed == expected_fixed, fixed


def test_sets():
    expected = {"a": "b", "b": "a"}
    test_dict = {"Set1": [[["a", "b"]]]}
    out = obfuscate.sample_obf(
        test_dict, perm_type="cycle", num_mappings=1, force_change=True
    )
    assert out[0] == expected


def test_tables():
    expected = {"a": "d", "b": "e", "c": "f", "d": "a", "e": "b", "f": "c"}
    test_dict = {"Table1": [[["a"], ["b"], ["c"]], [["d"], ["e"], ["f"]]]}
    out = obfuscate.sample_obf(
        test_dict, perm_type="cycle", num_mappings=1, force_change=True
    )
    assert out[0] == expected


def test_freetables():
    expected = [{"a": "d", "d": "a", "b": "f", "c": "e", "f": "b", "e": "c"}]
    test_dict = {"FreeTable1": [[["a"], ["b", "c"]], [["d"], ["e", "f"]]]}
    out = obfuscate.sample_obf(
        test_dict, perm_type="cycle", num_mappings=1, force_change=True
    )
    assert out[0] == expected[0]
