#!/usr/bin/env python
# coding: utf-8

import itertools
import json
import os
import pprint
import random
import re
import shutil
import traceback
import unicodedata as ud
import warnings
from functools import partial
from pathlib import Path

import fire
import numpy as np
import pandas as pd

# params
DATA_PATH = Path(__file__).resolve().parent.parent / "data"
ANN_PATH = DATA_PATH / "ann_puzzles"
RAW_PATH = DATA_PATH / "raw_puzzles"
OUTPUT_PATH = DATA_PATH / "obf/"
SHEET_PATH_MID = DATA_PATH / "Obfuscation - MID.csv"
SHEET_PATH_EASY = DATA_PATH / "Obfuscation - EASY.csv"
OBF_PATTERN = "@@@"
LANG_PATTERN = "$$$"
CULT_PATTERN = "&&&"
NUM_OBF = 6
SEED = 317
UNICODE_STD = "NFD"  # "NFKD"


CHAR_MAP = {
    "\u0294": "\u0242",
    "\u0241": "\u0242",
    "\\u0294": "\\u0242",
    "\\u0241": "\\u0242",
    "\\\\u0294": "\\\\u0242",
    "\\\\u0241": "\\\\u0242",
}


def get_fixed_terms(df, q_num):
    item = df[df["Num"] == q_num]["Fixed"].item()

    if pd.isna(item):
        return []
    terms = str(item).split(",")
    for t in terms:
        if t == "":
            print(
                f"Warning: Fixed - Problem: {q_num} -empty fixed term found in obfuscation rules for terms: {terms}"
            )
    return terms


def count_fixed(terms, s, ignore_case=True):
    res = {}
    if ignore_case:
        terms = [t.lower() for t in terms]
        s = s.lower()
    for t in terms:
        res[t] = s.count(t)
        if res[t] == 0:
            print(f"Debug: Fixed term with 0 count: {t}")
    return res


def no_multi_slashes(question, q_num, key):
    q = str(question)
    multi = []
    if "\\" in q:
        if key.lower() != "questions":
            multi.append("2")
    if "\\\\" in q:
        multi.append("3")
    if "\\\\" in q:
        multi.append("3")
    if "\\\\\\\\" in q:
        multi.append("4")
    if len(multi) > 0:
        print(
            f"Warning: Multi - Problem: {q_num} - multi-slash {', '.join(multi)} found in key: {key}"
        )
        return False
    return True


def not_identical(mappings: dict):
    for mapping in mappings:
        identical = True
        for k, v in mapping.items():
            if k != v:
                identical = False
        if identical:
            print(f"Warning: Identical character mapping was generated- {mapping}")
            return False
    return True


def clean_rules(rules):
    if not isinstance(rules, str):
        raise ValueError("Rules column should be string")

    mappings = CHAR_MAP.keys()

    rules = ud.normalize(UNICODE_STD, rules)

    for k in mappings:
        if k in rules:
            rules = rules.replace(k, CHAR_MAP[k])
            print(
                f"Info: Undeterministic char {k} found in Obfuscation Rules entry. Will be standardized to {rules}"
            )

    return rules


def clean_json(content, q_num=None):
    mappings = CHAR_MAP.keys()
    for key in [
        "preamble",
        "context",
    ]:
        if not isinstance(content[key], str):
            raise ValueError("Not string")
        else:
            for k in mappings:
                if k in content[key]:
                    print(
                        f"Info: Undeterministic char {k} found in {q_num} key: {key}. Will be standardized"
                    )
                    content[key] = content[key].replace(k, CHAR_MAP[k])
            content[key] = content[key].replace("\\", "")

    key = "questions"
    for k in mappings:
        if k in content[key]:
            print(
                f"Info: Undeterministic char {k} found in {q_num} key: {key}. Will be standardized"
            )
            content[key] = content[key].replace(k, CHAR_MAP[k])


def check_pattern(pattern, content):
    for c in content.values():
        if pattern in str(c):
            print(f"Found pattern: {c}")
            return True
    return False


def check_files(path, func, verbose=False):
    for r, d, files in os.walk(path):
        for f in files:
            if os.path.basename(f).split(".")[-1] == "json":
                f_path = os.path.join(r, f)
                if verbose:
                    print(f"Inspcting file: {f_path}")
                with open(f_path, "r", encoding="utf-8") as inp:
                    res = func(json.load(inp))
                    if res:
                        return True
    return False


def extract_pattern(q_content, pattern):
    res = []
    b = re.findall(rf"({pattern}(.|\n)*?{pattern})", q_content)
    if len(b) > 0:
        b = [x[0] for x in b]
    res.extend(b)
    return res


def retrieve_case(inp_string, cases):
    assert len(inp_string) == len(cases)
    res = []
    for l, c in zip(inp_string, cases):
        if c:
            res.append(l.upper())
        else:
            res.append(l)
    return "".join(res)


def replace_with_dictionary(a_string, dict_per, ignorecase=True, metadata=None):
    og_string = a_string

    if ignorecase:
        cases = [c.isupper() for c in a_string]
        a_string = a_string.lower()

    keys = list(dict_per.keys())
    index = 0
    new_string = []

    punctuation = [" ", ",", ";", "?", "-", "'", ".", "_"]
    while index < len(a_string):
        # Find longest matching key in mapping
        end = index + 1
        # Expand matching window While there's still possible keys to match:
        keys_to_match = set(keys)
        final_key = ""
        while end <= len(a_string) and len(keys_to_match) != 0:
            keys_loop = keys_to_match.copy()
            for k in keys_loop:
                if k[: end - index] == a_string[index:end]:
                    # check if we have a longer matching key than our current best match
                    if len(k) == (end - index) and len(k) > len(final_key):
                        final_key = k
                else:
                    # remove this key from candidate pool
                    keys_to_match = keys_to_match - set([k])
            end += 1

        if final_key == "":
            # no matching
            if a_string[index] not in punctuation:
                in_q = ""
                q_key = ""
                if metadata is not None:
                    in_q = metadata.get("q_num", None) or ""
                    q_key = metadata.get("key", None) or ""
                print(
                    f"Warning: Rules - Problem: {in_q} - No matching key in mapping for character {a_string[index]} - ord: {ord(a_string[index])}. Key: {q_key}. In mapping: {a_string[index] in keys}."
                )
                print(
                    f"Debug: Rules - Problem: {in_q} - cont. No matching in string {a_string}"
                )
                print(f"Debug: Rules - Problem: {in_q} - cont. No matching {keys}")

                # print(f"Whole string: {a_string}")
            match = a_string[index]
            step_size = 1
        else:
            match = dict_per[final_key]
            step_size = len(final_key)
            if metadata is not None and final_key in metadata["unused"]:
                metadata["unused"].remove(final_key)

        if cases[index]:
            match = match[0].upper() + match[1:]
            print(f"Debug: Case retrieved - {og_string[index]} -> {match}")

        new_string.append(match)
        index += step_size

    new_string = "".join(new_string)

    case_match = True
    if sum(cases) != sum([c.isupper() for c in new_string]):
        print(
            f"Warning: Case issue - {og_string} >>> {new_string}. num upper cases: {sum(cases)} vs. {sum([c.isupper() for c in new_string])}"
        )
        case_match = False

    print(a_string, end="")
    print(" >>> ", end="")
    print(new_string)

    return new_string, case_match


def obfuscate_text(
    text,
    mapping,
    pattern=OBF_PATTERN,
    metadata=None,
    ignorecase=True,
    unicode_escape=False,
    p_num="",
):
    bold_texts = extract_pattern(text, pattern)
    if len(bold_texts) == 0 and metadata["key"] != "preamble":
        print(
            f"Warning: Text does not contain any {pattern} pattern. nothing to obfuscate. Question: {metadata['q_num']} Key: {metadata['key']}"
        )
    obfuscated = text
    case_issues = 0
    for b in bold_texts:
        new_text = b
        # new_text = new_text.replace(pattern, "")
        new_text = remove_patterns(new_text, [pattern])
        if unicode_escape:
            new_text = new_text.encode("utf8").decode("unicode-escape")
        new_text = ud.normalize(UNICODE_STD, new_text)
        b_swapped, case_preserved = replace_with_dictionary(
            new_text, mapping, metadata=metadata, ignorecase=ignorecase
        )
        case_issues += int(not case_preserved)
        if unicode_escape:
            b_swapped = b_swapped.encode("utf8").decode()
            # b_swapped = b_swapped.replace("\\", "\\\\")

        obfuscated = obfuscated.replace(b, b_swapped)

    obfuscated = remove_patterns(obfuscated, [pattern, LANG_PATTERN, CULT_PATTERN])
    # print(obfuscated.encode('utf8').decode('unicode-escape'))
    return obfuscated, case_issues


def remove_patterns(text, patterns):
    res = text
    for p in patterns:
        res = res.replace(p, "")
    return res


def process_tables_dict(value):
    # First remove whitespace as these are often input variations
    value = value.replace(" ", "")
    value2 = value.replace("},{", "}//{").split("//")
    value2 = [x.replace("{", "").replace("}", "") for x in value2]
    value2 = [x.split(":") for x in value2]
    value3 = []
    for x in value2:
        value3.append([y.split(",") for y in x])
    # Check that the number of elements are as expected
    check_len_compatibility(value3)

    return value3


def create_obf_dictionary(
    df, q_num, colname="Num", exc="DecisionTable", unicode_normalize=True
):
    if exc:
        drop_cols = [x for x in df.columns.values if exc in x]
        df = df.drop(drop_cols, axis=1)

    df_sub = df[df[colname] == q_num]

    if df_sub.shape[0] > 1:
        print(
            f"Warning: {record.shape[0]} records matched for question: {q_num}. Taking the 1st one."
        )
    if df_sub.shape[0] == 0:
        raise ValueError(
            f"Warning: Rules - Problem: {q_num} - No obfuscation rules found"
        )

    table_cols = [x for x in df_sub.columns.values if "Table" in x]
    set_cols = [x for x in df_sub.columns.values if "Set" in x]
    fix_cols = [x for x in df_sub.columns.values if "Fixed" in x]

    tables_dict = {}

    # Unicode standardize
    if unicode_normalize:
        for col in table_cols + set_cols + fix_cols:
            if type(df_sub[col].item()) != float:
                df_sub.loc[:, col] = df_sub[col].apply(clean_rules)

    for col in table_cols:
        if type(df_sub[col].item()) != float:
            tables_dict[col] = process_tables_dict(df_sub[col].item())
    set_dict = {}

    for col in set_cols:
        if type(df_sub[col].item()) != float:
            set_dict[col] = [[df_sub[col].item().split(",")]]

    obf_dict = tables_dict | set_dict

    if str(df_sub["Fixed"].item()).lower() != "nan":
        fixed = df_sub["Fixed"].item().lower().split(",")
        obf_dict = obf_dict | {v: [[[v]]] for v in fixed}

    return obf_dict


def check_len_compatibility(values):
    all_row_len = []
    for column in values:
        row_len = []
        for row in column:
            row_len.append(len(row))
        all_row_len.append(row_len)

    # check that all elements of the list are identical
    assert all(x == all_row_len[0] for x in all_row_len)


def define_sub_permutations(num_items, perm_type, force_change=False):
    perm = []
    if perm_type == "random":
        perm = list(itertools.permutations(list(range(num_items))))
        if force_change:
            perm = perm[1:]
    elif perm_type == "cycle":
        perm = []
        if force_change:
            val_range = range(1, num_items)
        else:
            val_range = range(num_items)

        for increment in val_range:
            perm.append(((np.arange(num_items) + increment) % num_items))

    return perm


def find_perms(tables_dict, perm_type, force_change=False):
    all_perms = []
    for _, columns in tables_dict.items():
        if len(columns) > 1:
            outer_perm = define_sub_permutations(len(columns), perm_type, force_change)
        else:
            outer_perm = [[0]]

        inner_perms = []

        # as all columns should have the same number of elements in each row,
        # we only need to look at the number of elements in each row
        column = columns[0]
        for j in range(len(column)):
            row = column[j]
            if len(row) > 1:
                # Potentially force_change not required
                # this forces elements of row to permute
                inner_perm = define_sub_permutations(len(row), perm_type, force_change)
            else:
                inner_perm = [[0]]

            inner_perm = list(itertools.product(inner_perm, repeat=len(columns)))

            inner_perms.append(inner_perm)

        all_perms.append(list(itertools.product(outer_perm, *inner_perms)))

    return all_perms


def sample_obf(tables_dict, perm_type="cycle", num_mappings=1, force_change=False):
    perms = find_perms(tables_dict, perm_type, force_change)
    all_perms = list(itertools.product(*perms))

    num_sampled = min(num_mappings, len(all_perms))

    if len(all_perms) == 1 and not force_change:
        sampled_perms = all_perms[0]
        warnings.warn("There are no valid obfuscation for this question")
    else:
        sampled_perms = random.sample(all_perms, num_sampled)

    all_mappings = []
    for a_perm in sampled_perms:
        mappings = {}
        for ind in range(len(a_perm)):
            _, value = list(tables_dict.items())[ind]
            perm, inner_perm = a_perm[ind][0], a_perm[ind][1:]

            for i in range(len(value)):
                column = value[i]
                column_mapped = value[perm[i]]
                for j in range(len(column)):
                    row = column[j]
                    row_mapped = column_mapped[j]
                    dict_per = dict(zip(row, [row_mapped[x] for x in inner_perm[j][0]]))
                    mappings.update(dict_per)

                    if len(value) > 1:
                        dict_per2 = dict(
                            zip([row_mapped[x] for x in inner_perm[j][1]], row)
                        )
                        mappings.update(dict_per2)
        mappings = {
            ud.normalize(UNICODE_STD, k): ud.normalize(UNICODE_STD, v)
            for k, v in mappings.items()
        }
        all_mappings.append(mappings)
    return all_mappings


def valid_json(inp):
    try:
        json.loads(inp)
        return True
    except:
        return False


# Run obfuscation pipeline on all files under formatted_path and save in dest_path
def obf_files(
    formatted_path,
    dest_path,
    obf_df,
    num_samples,
    raw_path=RAW_PATH,
    break_on_exception=True,
):
    # Delete previous obfuscated files
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    os.makedirs(dest_path)

    # Loop through raw json files and generate obfuscated versions
    case_issues = 0
    processed = 0
    for root, d, files in os.walk(formatted_path):
        for f in sorted(files):
            if os.path.basename(f).split(".")[-1] == "json":
                print(f"Processing file: {f} " + "=" * 10)
                q_num = int(os.path.basename(f).split(".")[0][3:])
                print(f"Question number: {q_num}")
                f_path = os.path.join(root, f)
                q = json.load(open(f_path, "r", encoding="utf8"))

                # cleanup json
                clean_json(q, q_num)

                # Copy original file
                f_path_og = os.path.join(raw_path, str(q_num) + ".json")

                # Include original unobfuscated file to output
                # og_q = json.load(open(f_path_og, "r", encoding="utf8"))
                og_q = q.copy()

                for key in ["preamble", "context", "questions"]:
                    og_q[key] = remove_patterns(
                        og_q[key], patterns=[OBF_PATTERN, LANG_PATTERN, CULT_PATTERN]
                    )
                i = 0
                og_q["obfuscated"] = "False"
                og_q["obf_num"] = "0"
                og_q["mapping"] = None
                og_q["json_invalid"] = not valid_json(og_q["questions"])

                with open(os.path.join(dest_path, f"{q_num}_{i:04}.json"), "w") as out:
                    out.write(json.dumps(og_q, indent=4))
                    processed += 1

                try:
                    obf_dict = create_obf_dictionary(obf_df, q_num=q_num)
                    mappings = sample_obf(
                        obf_dict, num_mappings=num_samples, force_change=True
                    )
                    assert not_identical(mappings)
                    fixed_terms = get_fixed_terms(obf_df, q_num)

                except Exception as e:
                    print(f"Error processing file. Skipping.")
                    print(e)
                    traceback.print_exc()
                    if break_on_exception:
                        raise RuntimeError(
                            "Exiting due to error, to skip instead use break_on_exception=False"
                        )
                    continue

                for i, rep in enumerate(mappings):
                    print(f"Generated mapping: {i+1}")
                    print(f"Mapping: {rep}")
                    print("Obfuscated output:")
                    obfuscated_q = {}
                    metadata = {"q_num": q_num, "unused": list(mappings[0].keys())}
                    for key in ["preamble", "context", "questions"]:
                        # count fixed terms

                        metadata["key"] = key
                        if key == "questions":
                            og_string = (
                                str(q[key]).encode("utf8").decode("unicode-escape")
                            )
                            obfuscated, issues = obfuscate_text(
                                q[key],
                                mapping=rep,
                                metadata=metadata,
                                unicode_escape=True,
                                p_num=q_num,
                            )
                        else:
                            og_string = str(q[key])
                            obfuscated, issues = obfuscate_text(
                                q[key], mapping=rep, metadata=metadata, p_num=q_num
                            )

                        # Do some tests
                        if i == 0:
                            no_multi_slashes(og_string, q_num, key)

                        fixed_counts = count_fixed(fixed_terms, og_string)
                        if i == 0:
                            print(
                                f"Info: counts of fixed terms in origina file: {fixed_counts}"
                            )
                        # Test obfuscation is a valid json
                        obfuscated_q["json_invalid"] = not valid_json(obfuscated)

                        case_issues += issues
                        obfuscated_q[key] = obfuscated

                        # Check for weird escape chars
                        bad_escape = re.search(r"\\+[^utn\"\\]", obfuscated)
                        if bad_escape is not None:
                            bad_escape = bad_escape.group(0)
                            print(
                                f"Warning: OBF - Problem {q_num} - Found {bad_escape} in obfuscation {str(i + 1)} which might indicate wrong mapping"
                            )

                        # Ensure fixed terms are equal
                        fixed_counts_obf = count_fixed(fixed_terms, str(obfuscated))
                        if fixed_counts != fixed_counts_obf:
                            print(
                                f"Warning: Fixed - Problem: {q_num} - fixed terms counts mismatch - key: {key} Obf: {str(i+1)}"
                            )
                            print(
                                f"Debug: Fixed - Problem: {q_num} term counts in original - {key} : {fixed_counts}"
                            )
                            print(
                                f"Debug: Fixed - Problem: {q_num} term counts in obf - {key}: {fixed_counts_obf}"
                            )

                    obfuscated_q["obfuscated"] = "True"
                    obfuscated_q["obf_num"] = str(i + 1)
                    obfuscated_q["mapping"] = rep

                    if len(metadata["unused"]) > 0 and obfuscated is not None:
                        print(
                            f"Warning: Rules - Problem: {q_num} - unused phoneme(s) {metadata['unused']} found in obfuscated output."
                        )
                        for unused_c in metadata["unused"]:
                            print(
                                f"Debug: {unused_c} - ord: {[ord(c) for c in unused_c]}"
                            )

                    print(f"Number of letter casing with issues: {case_issues}")
                    with open(
                        os.path.join(dest_path, f"{q_num}_{i+1:04}.json"), "w"
                    ) as out:
                        out.write(json.dumps(obfuscated_q, indent=4))
                        processed += 1
    if processed == 0:
        print("Warning: no files were generated!")

    return processed


def pipeline(
    raw_questions_path: str = RAW_PATH,
    annotated_questions_path: str = ANN_PATH,
    sheet_path_easy: str = SHEET_PATH_EASY,
    sheet_path_mid: str = SHEET_PATH_MID,
    output_path: str = OUTPUT_PATH,
    num_obfuscations: int = NUM_OBF,
    pattern: str = OBF_PATTERN,
    random_seed: int = SEED,
):
    if num_obfuscations > 5:
        print(
            "Warning: in questions with few possible permutations, some obfuscations might be duplicate. Consider reducing number of obfuscations."
        )

    random.seed(random_seed)
    obf_df_mid = pd.read_csv(sheet_path_mid)
    obf_df_easy = pd.read_csv(sheet_path_easy)
    obf_df = pd.concat([obf_df_mid, obf_df_easy])

    assert obf_df.shape[0] > 0, "Empty obfuscation rules"

    print("Obfuscation sheet head:")
    print(obf_df.reset_index().head())

    # Ensuring that the pattern used in bold text extraction is not present in any of the raw json files:
    assert not check_files(
        raw_questions_path, partial(check_pattern, pattern), verbose=True
    ), "Failed: Annotation pattern found in raw files."

    assert not check_files(
        raw_questions_path, partial(check_pattern, LANG_PATTERN), verbose=True
    ), "Failed: Annotation pattern found in raw files."

    assert not check_files(
        raw_questions_path, partial(check_pattern, CULT_PATTERN), verbose=True
    ), "Failed: Annotation pattern found in raw files."

    # Run the pipeline and save output
    processed = obf_files(
        annotated_questions_path,
        output_path,
        obf_df,
        num_obfuscations,
        raw_questions_path,
    )

    if check_files(output_path, partial(check_pattern, pattern), verbose=True):
        print("Warning: Annotation pattern found in raw files.")

    if check_files(output_path, partial(check_pattern, LANG_PATTERN), verbose=True):
        print("Warning: Annotation pattern found in raw files.")

    if check_files(output_path, partial(check_pattern, CULT_PATTERN), verbose=True):
        print("Warning: Annotation pattern found in raw files.")

    print(f"{processed} obfuscated files were saved in {output_path}")
    print("Done!")


if __name__ == "__main__":
    fire.Fire(pipeline)
