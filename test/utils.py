testdata = [
    ("Ej AÀÁaã́ãà", "Ej AÀÁaã́ãà", {"E": "E"}, "No change"),
    ("", "", {"a": "b"}, "Handle empty input"),
    ("", "", {}, "Handle empty input and no mapping"),
]

swapping_data = [
    ("laikt", "LAikt", {"l": "L", "a": "A", "aif": "AIF"}, "Basic case"),
    ("nabxx", "nabxx", {"nothere": "NOTHERE"}, "No change: No matching keys"),
    ("aã́ã", "uṹũ", {"a": "u"}, "With diacritics"),
    ("aã́ãà", "ukũū", {"ã́": "k", "a": "u", "̀": "̄"}, "with more diacritics"),
    (
        "aã́ãà",
        "ukǔū",
        {"ã́": "k", "a": "u", "́": "̋", "̃": "̌", "̀": "̄"},
        "even more diacritics",
    ),
]


letter_case_data = [
    ("AÀÁaã́ãà", "KÙK̋kk̃̋k̃ù", {"a": "k", "à": "ù", "́": "̋"}, "upper case"),
    (
        "Ej AÀÁaã́ãà",
        "zzz kùk̋kk̃̋k̃ù",
        {"ej": "zzz", "a": "k", "à": "ù", "́": "̋"},
        "test upper case2 (obfuscated length changes",
    ),
    ("Čapital", "Ǩapital", {"c": "k"}, "capitalize 1st letter"),
]
