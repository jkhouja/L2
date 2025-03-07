import json
import os
from pathlib import Path

import fire
import load_questions
import pandas as pd
import pyminizip

"""
Dataset name in HF: "jkhouja/LingOly-TOO"
Run this script to convert data before uploading to huggingface"
When done, upload test_small.zip to huggingface dataset repo"
"""


# Paths to load
def publish(
    test_data_zip="../data/splits/benchmark_small.jsonl.zip",
    test_filename="../data/splits/benchmark.jsonl",
    test_filename_hf="../data/splits/test_small.jsonl",
    test_data_hf_zip="../data/splits/test_small.zip",
    PW="lingoly",
):
    # Loading benchmark
    print("loading benchmark")
    pyminizip.uncompress(test_data_zip, PW, "../data/splits", 0)
    os.chdir("../../code")

    question_sheets = []
    with open(test_filename) as f:
        for line in f:
            q = json.loads(line)
            question_sheets.append(q)
    Path.unlink(Path(test_filename))

    (
        questions,
        correct_answers,
        answers_train,
        q_idx,
        metadata,
    ) = load_questions.load_flattened_questions(
        question_sheets, model="o1", no_context=False, cot=False
    )
    print(f"Loaded {len(question_sheets)} items")

    # save prompts temp
    print(f"Saving to {test_data_hf_zip}")
    with open(test_filename_hf, "w") as f:
        for q, a, at, i, meta in zip(
            questions, correct_answers, answers_train, q_idx, metadata
        ):
            a = {
                x.encode("utf8")
                .decode("unicode-escape"): v.encode("utf8")
                .decode("unicode-escape")
                for x, v in a.items()
            }
            line = {
                "prompt": q,
                "completion": str(a),
                "question": meta["question_details"]["prompt"],
                "context": meta["question_details"]["metadata"]["context"],
                "overall_question_n": meta["question_details"]["metadata"][
                    "overall_question_n"
                ],
                "obfuscated_question_n": meta["question_details"]["metadata"][
                    "obfuscated_question_n"
                ],
                "question_n": i[-1],
                "obfuscated": meta["question_details"]["metadata"]["obfuscated"],
            }
            f.write(json.dumps(line) + "\n")

    # Save and compress
    pyminizip.compress(test_filename_hf, "", test_data_hf_zip, None, 0)
    Path.unlink(Path(test_filename_hf))
    print("Done")


if __name__ == "__main__":
    fire.Fire(publish)
