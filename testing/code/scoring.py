import argparse
import json
import os
import re
from pathlib import Path

import evaluate
import load_questions
import pandas as pd
import pyminizip
import scoring_methods
from tqdm import tqdm


def extract_json_substrings(input_string):
    """
    Extracts JSON-like substrings of dictionaries from a string input, even if incomplete.

    Args:
        input_string (str): The input string that may contain JSON-like dictionary substrings.

    Returns:
        list: A list of matched JSON-like substrings.
    """
    # Regex pattern to match JSON-like dictionary structures
    json_pattern = r"\{(?:[^{}]*|(?R))*}"  # Matches balanced curly braces

    def balanced_match(s):
        stack = []
        start = None
        matches = []

        for i, char in enumerate(s):
            if char == "{":
                if start is None:
                    start = i
                stack.append("{")
            elif char == "}" and stack:
                stack.pop()
                if not stack:
                    matches.append(s[start : i + 1])
                    start = None

        # If the string ends with an unbalanced part, try to include it
        if stack and start is not None:
            matches.append(s[start:] + "}")

        if len(matches) == 1:
            try:
                cont = matches[0].replace("'", '\\"')
                # print(cont)
                return json.loads(cont)
            except:
                try:
                    return eval(matches[0])
                except:
                    return matches[0]

        elif len(matches) == 0:
            return ""
        else:
            return matches

    if input_string is None:
        return ""
    matches = balanced_match(input_string)
    return matches


def clean_key(key):
    for c in [" ", ".", ")", "("]:
        key = key.replace(c, "")
    if key == "":
        print(f"Warning: Erased key: {key}")
    return key


def find_match(ans_items, key):
    key = clean_key(str(key).strip())

    if isinstance(ans_items, list):
        for ans_dict in ans_items:
            if not isinstance(ans_dict, dict):
                try:
                    ans_dict = json.loads(ans_dict)
                except:
                    print(f"Parsing issue of {ans_dict}")
                    continue

            ans_dict = {clean_key(str(k)): v for k, v in ans_dict.items()}
            retr = str(ans_dict.get(key, ""))
            if retr != "":
                print("Found item from json")
                return retr

    elif isinstance(ans_items, dict):
        ans_items = {clean_key(str(k)): v for k, v in ans_items.items()}
        retr = str(ans_items.get(key, ""))
        if retr != "":
            print("Found item from json")
            return retr
    else:
        try:
            ans_dict = json.loads(ans_items)
            ans_dict = {clean_key(str(k)): v for k, v in ans_dict.items()}
            retr = str(ans_dict.get(key, ""))
            if retr != "":
                print("Found item from json")
                return retr
        except:
            print(f"Parsing issue of {ans_dict}")

    return ""


def extract_answers(entry):
    extracted = entry["model_answers_extracted"]
    try:
        # extracted = json.loads(extracted)
        extracted = eval(extracted)
    except:
        print(f"Failed to parse extracted: {extracted}")
        extracted = entry["model_answers_extracted"]
    model_answers = entry["model_answers"]

    out = {}
    for k, v in model_answers.items():
        if v is not None and v != "":
            # Use original model response if available
            print(f"Using original answer: {v}")
            if "IMPROPER PARSING:" in str(v):
                try:
                    if extracted != "" and extracted is not None:
                        ans_items = extracted
                    else:
                        ans_items = extract_json_substrings(str(v))
                    print(f"Extracted {ans_items}")
                    out[k] = find_match(ans_items, k)
                    if out[k] == "" and ans_items != "":
                        print(f"Failed to find key: {k} in {ans_items}")

                except:
                    # Just use original reponse
                    print(f"Exception parsing Improper for key: {k}")
                    print(ans_items)
                    out[k] = v
            else:
                out[k] = v
        else:
            print(
                f"Trying to use extracted answer: {extracted} type: {type(extracted)}"
            )
            if isinstance(extracted, dict):
                out[k] = find_match(extracted, k)
                print(f"Found {out[k]} from {extracted}")
            else:
                print("No json in raw data and no model response. returning empty")
                out[k] = ""

    return out


def listtostr(l):
    if isinstance(l, list):
        s = ""
        for i in l:
            s += i + ", "
        s = s[:-2]
        return s
    else:
        return l


if __name__ == "__main__":
    #########################
    ##### Main pipeline #####
    #########################
    tqdm.pandas()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="The name of the model to run. (See registry dict)"
    )

    parser.add_argument("--no_context", type=bool, default=False)
    parser.add_argument("--cot", type=bool, default=False)
    parser.add_argument("--add_formatter", type=bool, default=True)

    parser.add_argument(
        "--outfolder",
        type=str,
        help="The folder filepath to save the results.",
        default="../data/scores/",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        help="The filename to save the results.",
        default="modelname_questionsname.json",
    )
    parser.add_argument(
        "--responsesfolder",
        type=str,
        help="The path of the responses to score",
        default="../data/responses_obf/",
    )
    parser.add_argument(
        "--harness",
        type=bool,
        default=False,
    )

    args = parser.parse_args()

    model = args.model
    no_context = args.no_context
    cot = args.cot
    harness = args.harness
    add_formatter = args.add_formatter

    if no_context:
        print(f"Running in no context mode")
    if cot:
        print(f"Running in chain of thought mode")

    # model registry
    with open("../data/model_list.json") as f:
        model_list = json.load(f)

    model_name = model_list[model].get("chkpoint", None) or model_list[model]["name"]

    if args.outfile == "modelname_questionsname.json":
        if cot:
            if harness:
                filename = f"{model_name.split('/')[-1]}_harness_lingoly_cot"
            else:
                filename = f"{model_name.split('/')[-1]}_lingoly_cot"

        elif no_context:
            if harness:
                filename = f"{model_name.split('/')[-1]}_harness_lingoly_nocontext"
            else:
                filename = f"{model_name.split('/')[-1]}_lingoly_nocontext"
        else:
            if harness:
                filename = f"{model_name.split('/')[-1]}_harness_lingoly"
            else:
                filename = f"{model_name.split('/')[-1]}_lingoly"
    else:
        filename = args.outfile

    print(f"Loading model respones from: {filename}")
    with open(args.responsesfolder + filename + ".json") as f:
        responses = json.load(f)

    responses = pd.DataFrame(responses)
    responses.set_index("obfuscated_question_n", inplace=True)

    # responses = responses[:500]

    print(f"Loaded: {responses.shape}")

    # Strip the answer from the 1 element list
    responses["correct_answers"] = responses["correct_answers"].apply(lambda x: x[0])

    # reparse model_answer from json
    responses["model_answers_original"] = responses["model_answers"]
    responses["model_answers_extracted"] = responses["model_raw_response"].map(
        extract_json_substrings
    )
    responses["model_answers"] = responses.apply(extract_answers, axis=1)

    # loading these is the slowest part of scoring, so passing them in avoids loading every time
    bleu = evaluate.load("bleu")
    # rouge = evaluate.load("rouge")
    chrf = evaluate.load("chrf")

    responses["total_parts"] = responses["correct_answers"].apply(len)

    print(f"Scoring Exact Match")
    responses["exact_match_score"] = responses.progress_apply(
        lambda x: scoring_methods.compute_scores(
            x, scoring_methods.safe_exact, helper=None
        ),
        axis=1,
    )

    print(f"Scoring Bleu")
    responses["bleu_score"] = responses.progress_apply(
        lambda x: scoring_methods.compute_scores(x, scoring_methods.safe_bleu, bleu),
        axis=1,
    )

    # print(f"Scoring Rouge")
    # responses["rouge_score"] = responses.progress_apply(
    #    lambda x: scoring_methods.compute_scores(x, scoring_methods.safe_rouge, rouge),
    #    axis=1,
    # )

    print(f"Scoring Char F")
    responses["chrf_score"] = responses.progress_apply(
        lambda x: scoring_methods.compute_scores(x, scoring_methods.safe_chrf, chrf),
        axis=1,
    )

    # responses["key"] = [m["key"] for m in metadata]

    # responses.drop("correct_answers", axis=1, inplace=True)

    print(f"Saving to {filename}.csv")
    responses.to_csv(args.outfolder + filename + ".csv")
    print(f"Done")
