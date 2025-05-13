import json
import os
import random
import time
import traceback
from pathlib import Path
from typing import List

import fire
import load_questions
import prompt_models
import pyminizip
import tiktoken
import torch
from guidance import models
from tqdm import tqdm

# os.environ["HF_HOME"] = "YOUR_HF_HOME"
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MAX_API_ATTEMPT = 10
OUTFOLDER = "../data/responses_obf"
TMP_PATH = "../data/responses_obf/tmp"
MODEL_LIST = "../data/model_list.json"
PW = "lingoly"

FILTER_PROB = [69, 164]
hf_token = os.getenv("HF_TOKEN")

ENCODING = "cl100k_base"
encoding = tiktoken.get_encoding(ENCODING)

manipulation_strategies = [
    "Here's an exciting task {} \n Now Answer \n ",
    "Here's an exciting task {}",
]


# From OpenAI cookbook. using one ecoding as an estimate only
def get_tokens_count(q: str) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(q))
    return num_tokens


def load_cache(model_name, tmp_path=TMP_PATH, suffix="_lingoly"):
    cached = []
    cached_dict = {}
    for root, dirs, files in os.walk(tmp_path):
        for file in files:
            # print(file)
            if model_name + suffix in file and "tmp" in file:
                # add this to list of cached files
                cached.extend(json.load(open(Path(root, file), "r")))
    for entry in cached:
        prompt_id = entry["obfuscated_question_n"] + entry["question_n"]
        if prompt_id in cached_dict:
            raise ValueError(f"Duplicate keys: {prompt_id}")
        cached_dict[prompt_id] = entry
    return cached_dict


def manipulate_prompts(prompts: List[str], metadata: dict):
    outs = []
    for p, meta in zip(prompts, metadata):
        strategy_idx = random.randint(0, len(manipulation_strategies) - 1)
        edited = manipulation_strategies[strategy_idx].format(p)
        edited = edited.replace("\n\n", "\n\n\n")
        edited = edited.replace("  ", "   ")
        chars = meta["question_details"]["metadata"]["mapping"]
        if chars is not None:
            chars = chars.values()
        else:
            chars = ""
        edited = edited + "\n" + "".join(chars)
        outs.append(edited)

    return outs


def make_batch(questions, answers, indices, keys, batch_size=1, adversarial=False):
    batches = []
    for i in range(0, len(questions), batch_size):
        qs = questions[i : i + batch_size]
        if adversarial:
            qs = manipulate_prompts(qs, keys[i : i + batch_size])
        batches.append(
            {
                "questions": qs,
                "answers": answers[i : i + batch_size],
                "index": indices[i : i + batch_size],
                "metadata": keys[i : i + batch_size],
            }
        )

    return batches


##################################
## Actual running code
##################################


def pipeline(
    model: str,
    model_list_path: str = MODEL_LIST,
    test_data_zip: str = "../data/splits/benchmark.jsonl.zip",
    test_filename: str = "../data/tmp/benchmark.jsonl",
    tmp_dir: str = "../data/tmp/",
    pw: str = PW,
    cot: bool = False,
    device_map: str = "cuda",
    no_context: bool = False,
    questions_limit: int = None,
    outfolder: str = OUTFOLDER,
    generate_only: bool = False,
    outfile: str = "modelname_questionsname.json",
    adversarial: bool = False,
    use_cache: bool = True,
    repeats: int = 1,
    custom_prompt: bool = False,
    filter_prob: bool = False,
):
    # Make sure we don't use cache when repeats > 1
    if repeats > 1 and use_cache:
        raise ValueError("Cannot use cache when repeats > 1")

    global FILTER_PROB
    if not filter_prob:
        FILTER_PROB = []
        print("Problems filtering disabled")

    device_map = device_map
    if device_map.isnumeric():
        device_map = int(device_map)

    if no_context:
        print(f"Running in no context mode")
    if cot:
        print(f"Running in chain of thought mode")
    if adversarial:
        print("Running in adversarial mode")
    if repeats > 1:
        print(f"Will repeat each prompt {repeats} times")

    # Checking model in model list
    # model registry
    with open(model_list_path) as f:
        model_list = json.load(f)

    model_details = model_list[model]
    model_name = model_details["name"]
    checkpoint_name = model_details.get("chkpoint", None)

    # Loading data
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    print(f"Loading questions. Limited to {questions_limit}")
    pyminizip.uncompress(test_data_zip, pw, tmp_dir, 0)
    os.chdir("../../code")
    question_sheets = []
    with open(test_filename) as f:
        for line in f:
            q = json.loads(line)
            if len(FILTER_PROB) == 0 or q["index"][0] in FILTER_PROB:
                question_sheets.append(q)
    Path.unlink(Path(test_filename))

    (
        questions,
        correct_answers,
        answers_train,
        q_idx,
        metadata,
    ) = load_questions.load_flattened_questions(
        question_sheets,
        model=model,
        no_context=no_context,
        cot=cot,
        custom_prompt=custom_prompt,
    )

    if outfile == "modelname_questionsname.json":
        nc = ""
        ct = ""
        adv = ""
        rp = ""
        cust = ""
        if no_context:
            nc = "_nocontext"
        if cot:
            ct = "_cot"
        if adversarial:
            adv = "_adv"
        if repeats > 1:
            rp = f"_rp{repeats}"
        if custom_prompt:
            cust = "_customprompt"
        model_name_str = model_name
        if checkpoint_name:
            model_name_str = checkpoint_name
        filename = (
            f"{model_name_str.split('/')[-1]}_lingoly{cust}{nc}{ct}{adv}{rp}.json"
        )
    else:
        filename = outfile

    # save prompts temp
    tmp_file = f"{test_filename.split('.jsonl')[-2]}_prompts.jsonl"
    print(tmp_file)
    max_prompt_len = 0
    total_tokens = 0
    with open(tmp_file, "w") as f:
        for q, a, at, i in zip(questions, correct_answers, answers_train, q_idx):
            length_q = get_tokens_count(q)
            total_tokens += length_q
            max_prompt_len = max(max_prompt_len, length_q)
            line = {"prompt": q, "completion": a, "completion_train": at, "index": i}
            f.write(json.dumps(line) + "\n")

    print(f"Data size: {len(question_sheets)}")
    print(
        f"Maximum prompt length in chars: {max_prompt_len} or ~{max_prompt_len/4} tokens. Total (chars or tokens): {total_tokens}"
    )

    cached_dict = None
    if use_cache:
        if outfile == "modelname_questionsname.json":
            model_tmp_name = model_name
            if checkpoint_name:
                model_tmp_name = checkpoint_name
            model_tmp_name = model_tmp_name.split("/")[-1]
            model_tmp_name = f"{model_tmp_name}{cust}{nc}{ct}{adv}{rp}"
        else:
            model_tmp_name = filename
        model_tmp_name = filename
        print(f"Looking for cache for {model_tmp_name}")
        cached_dict = load_cache(f"{model_tmp_name}", suffix="")

        print(f"found {len(cached_dict)} cached responses")

    if generate_only:
        print("Running in generation mode only. Finished.")
        return

    if questions_limit is None:
        questions_limit = len(questions)
    questions, correct_answers, q_idx = (
        questions[:questions_limit],
        correct_answers[:questions_limit],
        q_idx[:questions_limit],
    )

    ## Loading the model
    print("Loading model")

    if model_details["model_type"] == "guidance":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            # trust_remote_code=True,
            use_fast=False,
        )
        tokenizer.pad_token = tokenizer.eos_token

        # required for using guidance
        if model in ["Llama_3_8B", "Llama_3_70B"]:
            with open("../data/llama3_decoder.json", "r") as f:
                byte_decoder = json.load(f)
                tokenizer.byte_decoder = byte_decoder

        if model_details["dtype"] == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=False,
            )

        lm = models.Transformers(
            model_name,
            tokenizer=tokenizer,
            echo=False,
            quantization_config=quantization_config,
            device_map={"": device_map},
        )

    if model_details["model_type"] == "transformers":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            force_download=False,
            # trust_remote_code=True,
            use_fast=True,
        )
        tokenizer.pad_token = tokenizer.eos_token

        # required for using guidance
        if model in ["Llama_3_8B", "Llama_3_70B"]:
            with open("../data/llama3_decoder.json", "r") as f:
                byte_decoder = json.load(f)
                tokenizer.byte_decoder = byte_decoder

        if model_details["dtype"] == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=False, bnb_4bit_compute_dtype=torch.float16
            )
            quantization_config = None

        lm = AutoModelForCausalLM.from_pretrained(
            checkpoint_name or model_name,
            force_download=False,
            token=hf_token,
            device_map=device_map,
            # use_flash_attention_2=True,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
        lm.eval()

    # Running predictions
    print("Predicting on quesitons")
    i = 0
    qas = []
    err = []
    for batch in tqdm(
        make_batch(
            questions,
            correct_answers,
            q_idx,
            metadata,
            batch_size=1,
            adversarial=adversarial,
        )
    ):
        retrieved = False
        attempt = 0
        prompt_id = batch["index"][0][1] + batch["index"][0][4]
        if (
            use_cache
            and len(cached_dict) > 0
            and cached_dict.get(prompt_id, None) is not None
        ):
            rs = cached_dict[prompt_id]
            if rs["questions"] != batch["questions"][0] and not adversarial:
                raise ValueError(
                    f"Cached question does not match Data question: {prompt_id}"
                )
            retrieved = True
            print(f"Using cached for {i}: {prompt_id}")
            qas.append(rs)
        else:
            # Run prediction
            repetition = 0
            while attempt <= MAX_API_ATTEMPT and repetition < repeats:
                try:
                    if model_details["model_type"] == "guidance":
                        (
                            responses,
                            raw_output,
                        ) = prompt_models.guidance_prompt_open_model(
                            lm, batch, tokenizer, cot
                        )
                    elif model_details["model_type"] == "transformers":
                        if model_details.get("as_messages", False):
                            call_f = prompt_models.transformers_prompt_llama_model
                        else:
                            call_f = prompt_models.transformers_prompt_open_model

                        responses, raw_output = call_f(
                            lm, batch, tokenizer, device_map, cot
                        )
                    else:
                        responses, raw_output = prompt_models.prompt_closed_model(
                            batch, model_details, cot
                        )

                    rs = {
                        "questions": batch["questions"][0],
                        "split_key": batch["metadata"][0]["split_key"],
                        "overall_question_n": batch["index"][0][0],
                        # "subquestion": batch
                        "obfuscated_question_n": batch["index"][0][1],
                        "correct_answers": batch["answers"],
                        "obfuscated": batch["index"][0][2],
                        "obf_num": batch["index"][0][3],
                        "question_n": batch["index"][0][4],
                        "model_answers": responses,
                        "model_raw_response": raw_output,
                        "repetition_idx": repetition,
                    }
                    repetition += 1
                    qas.append(rs)
                except:
                    print(f"Error in API call in attempt {attempt}.", end=" ")
                    traceback.print_exc()
                    attempt += 1
                    if attempt == MAX_API_ATTEMPT:
                        print()
                        raise ValueError("Maximum attempts reached")
                    else:
                        sleep_amnt = 10
                        print(f"Trying again in {sleep_amnt}")
                        time.sleep(sleep_amnt)

        # saving along the way just in case
        i += 1
        step_size = 2
        if i % step_size == 0 and not retrieved:
            folder = Path(outfolder + "/tmp")
            folder.mkdir(exist_ok=True)
            try:
                (folder / f"{filename}_tmp{i}").write_text(
                    json.dumps(qas[-step_size * repeats :], indent=4)
                )
            except:
                print(f"Error writing json at step {i}. Will discard last two entries")
                err.append(qas[-step_size:])
                qas = qas[:-step_size]

    folder = Path(outfolder)
    folder.mkdir(exist_ok=True)
    # output if possible
    try:
        (folder / filename).write_text(json.dumps(qas, indent=4))
    except:
        print(f"Error writing final output json. Will attempt to one example at a time")
        with open((folder / filename), "w") as f_out:
            f_out.write("[\n")
            for e in qas:
                try:
                    f_out.write(json.dumps(e, indent=4))
                except:
                    continue

            f_out.write("\n]")

    # output errors
    err_filename = filename + "_err"
    with open((folder / err_filename), "w") as err_out:
        for e in err:
            err_out.write(json.dumps({"entry": str(e)}) + "\n")


if __name__ == "__main__":
    fire.Fire(pipeline)
