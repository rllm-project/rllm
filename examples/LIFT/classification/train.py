# Paper: LIFT: Language-Interfaced FineTuning for Non-Language Machine Learning Tasks (NeurIPS 2022)
# macro_f1: 0.1293655952583329
# micro_f1: 0.29685807150595883
# Total time: 875.0060040950775s


import pandas as pd
from functools import partial
import sys
import re
import argparse
import time

sys.path.append("../src")
from utils.helper import data2text,write_jsonl
import models.lora_gptj as GPTJ
# from run_exps_helper import *
import torch
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.append("../../../")
from rllm.utils import macro_f1_score, micro_f1_score, get_llm_chat_cost

time_start = time.time()

def data2text(row, label = True, init = '', end = ''):
    prompt = init 
    prompt += ' Title:'+str(row['Title']).replace("'", "").replace('"', '')\
        # +' Director:'+str(row['Director']).replace("'", "").replace('"', '')\
        # +' Cast:'+str(row['Cast'])+' Runtime:'+str(row['Runtime']).replace("'", "").replace('"', '')\
        # +' Plot:'+str(row['Plot']).replace("'", "").replace('"', '')
        # +' Languages:'+ str(row['Languages']).replace("'", "").replace('"', '')\
        # +' Certificate:'+str(row['Certificate']).replace("'", "").replace('"', '')\
        # +' Year:'+ str(row['Year']).replace("'", "").replace('"', '')\
    prompt += end

    if not label:
        # final_prompt = "{\"prompt\":\"%s###\", \"completion\":\"@@@\"}" % (prompt)
        final_prompt = "{\"prompt\":\"%s###\"}" % (prompt)
    else:
        completion = row['Genre']
        final_prompt = "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)
    return final_prompt

def df2propmts(df, data2text_func, init='', end='', label=True):
    jsonl = df.apply(lambda row: data2text_func(row, label=label, init=init, end=end), axis=1).tolist()
    return jsonl

parser = argparse.ArgumentParser(description='')
parser.add_argument("-g", "--gpu_id", default=0, type=int)
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--seed", default=12345, type=int)
parser.add_argument("-p", "--is_permuted", action="store_true")

parser.add_argument("-v", "--eval", default=0, type=int)
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu_id}') if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(args.gpu_id)

users = pd.read_csv(
    '../../../rllm/datasets/rel-movielens1m/classification/users.csv')
train = pd.read_csv(
    '../../../rllm/datasets/rel-movielens1m/classification/movies/train.csv')
val = pd.read_csv(
    '../../../rllm/datasets/rel-movielens1m/classification/movies/validation.csv')
test = pd.read_csv(
    '../../../rllm/datasets/rel-movielens1m/classification/movies/test.csv')
ratings = pd.read_csv(
    '../../../rllm/datasets/rel-movielens1m/classification/ratings.csv')

init='Given information about a movie: '
end = 'What is the genres it may belong to? Note: 1. Give the answer as following format: genre_1|genre_2|...|genre_n 2. The answers must only be chosen from followings:Documentary, Adventure, Comedy, Horror, War, Sci-Fi, Drama, Mystery, Western, Action, Children\'s, Musical, Thriller, Crime, Film-Noir, Romance, Animation, Fantasy'

train_prompts = df2propmts(train, data2text, init, end)
val_prompts = df2propmts(val, data2text, init, end)
test_prompts = df2propmts(test, data2text, init, end)


write_jsonl('\n'.join(train_prompts),'train.json')
write_jsonl('\n'.join(val_prompts),'val.json')
write_jsonl('\n'.join(test_prompts),'test.json')

y_val = val['Genre']
y_test = test['Genre']

movie_genres = test["Genre"].str.split("|")
# print(type(test["Genre"]))
# print(movie_genres)
all_genres = list(set([genre for genres in movie_genres for genre in genres]))
print(y_test)

# gpt = GPTJ.LoRaQGPTJ(adapter=True, device=device,model_name='hivemind/gpt-j-6B-8bit')

gpt = GPTJ.LoRaQGPTJ(adapter=True, device=device)
train_configs={'learning_rate': 1e-5, 'batch_size': 2, 'epochs':1,  'weight_decay': 0.01, 'warmup_steps': 6}
gpt.finetune('data/train.json', 'data/val.json', train_configs, saving_checkpoint=False)

test_prompts = extract_prompts('data/test.json')
pred = query(gpt, test_prompts,bs=8)
# write_jsonl('\n'.join(pred),'pred.json')
# print(pred)

# pred = pd.DataFrame({'Genre':pred})
# y_pred = pred['Genre'].str.split("|")
# y_pred_filtered = []
# for genres in y_pred:
#     filtered_genres = [genre for genre in genres if genre in all_genres]
#     if len(filtered_genres) == 0:
#         y_pred_filtered.append(pd.Series([]))
#     else:
#         y_pred_filtered.append(pd.Series(filtered_genres))
y_pred = []

for row in pred:
    filter_row = []
    if row:
        split_row = row.split('|')
        for genre in split_row:
            if genre in all_genres: filter_row.append(genre)
    y_pred.append(filter_row)
    
y_pred=pd.Series(y_pred)
    




mlb = MultiLabelBinarizer(classes=all_genres)
real_genres_matrix = mlb.fit_transform(movie_genres)
# print(real_genres_matrix)
pred_genres_matrix = mlb.fit_transform(y_pred)
# print(pred_genres_matrix)
macro_f1 = macro_f1_score(real_genres_matrix, pred_genres_matrix)
micro_f1 = micro_f1_score(real_genres_matrix, pred_genres_matrix)

time_end = time.time()

print(f"macro_f1: {macro_f1}")
print(f"micro_f1: {micro_f1}")
print(f"Total time: {time_end - time_start}s")
# print(f"Total USD$: {total_cost}")