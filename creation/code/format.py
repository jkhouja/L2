import json
import os
import traceback

import fire


def format_dir(path):
    for r, d, files in os.walk(path):
        for f in files:
            if f.split(".")[-1] == "json":
                try:
                    with open(r + f, "r") as inpt:
                        print(f"Processing file: {f}")
                        res = []
                        c = json.loads(inpt.read())
                        for k, v in c.items():
                            res.append(k)
                            res.append("------")
                            if k == "questions":
                                sc = json.loads(v)
                                for items in sc:
                                    for sk, sv in items.items():
                                        if sk == "subprompts":
                                            res.append(sk)
                                            for sitems in sv:
                                                for ssk, ssv in sitems.items():
                                                    s = f"\t {ssk}: \t {ssv}"
                                                    res.append(s)
                                                res.append("\n")
                                        else:
                                            s = f"{sk}: \t {sv}"
                                            res.append(s)
                            else:
                                res.append(str(v))
                            res.append("\n")
                        res = "\n".join(res)
                        with open(r + "formatted/" + f + ".txt", "w") as outf:
                            outf.write(res)
                except:
                    traceback.print_exc()
                    print(f"Error formatting file: {f}, skpping")


def main(path="../data/obf/"):
    try:
        os.mkdir(path + "/formatted")
        print(f"Directory '{path}' created.")
    except FileExistsError:
        print("Directory already exist")
    format_dir(path)


if __name__ == "__main__":
    fire.Fire(main)
