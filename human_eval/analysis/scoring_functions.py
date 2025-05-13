import pandas as pd
import json
import numpy as np
import os
import ast
import unicodedata as ud
import re
import evaluate

def find_file(filename):
    """ Locate the relevant json file """
    directories = ['../data/unobfuscated', '../data/obfuscated_1', '../data/obfuscated_2']
    for directory in directories:
        # Check if the file exists in the current directory
        full_path = os.path.join(directory, filename)
        if os.path.isfile(full_path):

            with open(full_path, 'r') as file:
                return json.load(file)

def clean_answer(answer: str):
    """ Taken from original LingOly code -> added rule for 'r' """
    # make all answers strings
    answer = str(answer)

    # remove whitespace and final stop
    clean = str(answer.strip()).strip(".")

    # reduce multiple spaces to a single space
    clean = re.sub(r"[ ]+", " ", clean)

    # reduce to lower case
    clean = clean.lower()

    # remove internal + (can't currently handle for marking)
    clean = re.sub("\\+", "", clean)

    # make quotes consistent
    quotes_map = {"‘": "'", "’": "'", "’": "'", "“": '"', "”": '"'}

    for k, v in quotes_map.items():
        clean = re.sub(k, v, clean)

    # make unicode consistent
    clean = ud.normalize("NFKD", clean)

    # replace ɾ with r 
    clean = clean.replace('\u027e', 'r')

    return clean
            
# get answers -> return in dict
def extract_answers(json_, prob_code):
    answers = {}
    count = 0
    for i in json.loads(json_['questions']):
        for j in i['subprompts']:
            count+=1
            answers.update({prob_code+f"_{count}":j['answer']})
            
    return answers

# chrf set up
chrf = evaluate.load("chrf")

# score funct
def score(row, dict_, verbose = 0):
    points = 0
    max_points = 0
    Ligaurian_warning = False
    isna_count = 0
    chrf_l = []
    for i in dict_.keys():
        max_points += 1

        if row[i] is np.nan:
            isna_count += 1
            chrf_l.append(0)
            continue
        
        try:
            if type(ast.literal_eval(dict_[i])) == list:
                answer_list = [clean_answer(x) for x in ast.literal_eval(dict_[i])]
                if clean_answer(row[i]) in answer_list:
                    points += 1
                chrf_scores = [chrf.compute(references=[x], predictions=[clean_answer(row[i])])['score'] for x in answer_list]
                chrf_l.append(chrf_scores.max())
        except: 
            # allows some formatting errors for Ligurian
            if (i[0:3]=='147') and (int(i[-1])>=2):
                if clean_answer(row[i]).split(" ")[0] == clean_answer(dict_[i]):
                    Ligaurian_warning = True
                    points += 1

            elif clean_answer(row[i]) == clean_answer(dict_[i]):
                points += 1

            elif (i[0:3]=='160') and (i[-1]=='5'):
                if clean_answer(row[i])[0] == clean_answer(dict_[i]): 
                    points += 1

            chrf_l.append(chrf.compute(references=[clean_answer(row[i])], predictions=[clean_answer(dict_[i])])['score'])

    if verbose > 0:
        if Ligaurian_warning == True: print(f"Ligurian Warning")

    return isna_count, points, max_points, (points/max_points)*100, np.mean(chrf_l)

def row_wise_scoring(row, df):
    """ Application of the scoring function to specific rows in the df """

    # if no code then return nan
    if pd.isna(row['problem_code']):
        return np.nan, np.nan, np.nan, np.nan

    # load answer json and score
    json_ = find_file(row['problem_code']+'.json')
    answ_dict = extract_answers(json_, row['problem_code'])

    na, p, max_p, sc, chrf = score(row, answ_dict, verbose=0)

    return na, p, max_p, sc, chrf