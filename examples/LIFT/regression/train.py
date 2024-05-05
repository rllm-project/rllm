# Paper: LIFT: Language-Interfaced FineTuning for Non-Language Machine Learning Tasks (NeurIPS 2022)
# mae_loss: 1.044205298013245
# Total time: 4518.223552465439s



import pandas as pd
from functools import partial
import sys
import time

sys.path.append("../src")
from utils.helper import data2text,write_jsonl
import models.lora_gptj as GPTJ
from run_exps_helper import *
import torch
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.append("../../../")
from rllm.utils import mae, get_llm_chat_cost

time_start = time.time()

def df2prompts(df:pd.DataFrame, init = '',end = '',prompts_each_user = 1, n_given_rows = 5,sample_users=100,label =True):
    grouped = df.groupby('UserID')
    if sample_users:
        selected_users = grouped['UserID'].unique().sample(n=sample_users, replace=False)
        # print(selected_users)
        selected_groups = [grouped.get_group(user[0]) for user in selected_users]
        # print(selected_groups)
    else:
        selected_groups = [group for _, group in grouped]
        
    jsonl = []
    for group in selected_groups:
        
        # print(group)
        for i in range(prompts_each_user):
            given_rows = group.sample(n = n_given_rows,replace= True )
            infer_rows = group.sample(n = 1,replace= True)
            prompt = init 
            
            n = 0
            for index,row in given_rows.iterrows():
                n += 1
                id = row['MovieID']
                movie_info = movies[movies["MovielensID"] == id]
                prompt += str(n)+') '\
                    'Title: ' + str(movie_info['Title'].values[0]).replace("'", "").replace('"', '') + ' ' \
                    'Genre: ' + str(movie_info['Genre'].values[0]).replace("'", "").replace('"', '') + ' ' \
                    'Rating: '+ str(row['Rating']).replace("'", "").replace('"', '')+'; '
            
            prompt += 'Now I want you to predict the user\'s ratings for the following movie(s): '
            
            n = 0
            for index,row in infer_rows.iterrows():
                n += 1
                id = row['MovieID']
                prompt += \
                    'Title: ' + str(movie_info['Title'].values[0]).replace("'", "").replace('"', '') + ' ' \
                    'Genre: ' + str(movie_info['Genre'].values[0]).replace("'", "").replace('"', '') + '; ' \

            prompt += end
            if label:
                completion = "|".join([str(row['Rating']) for index,row in infer_rows.iterrows()])
                final_prompt = "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)
            else:
                final_prompt = f"{prompt}###"
            jsonl.append(final_prompt)
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
    '../../../rllm/datasets/rel-movielens1m/regression/users.csv')
train = pd.read_csv(
    '../../../rllm/datasets/rel-movielens1m/regression/ratings/train.csv')
val = pd.read_csv(
    '../../../rllm/datasets/rel-movielens1m/regression/ratings/validation.csv')
test = pd.read_csv(
    '../../../rllm/datasets/rel-movielens1m/regression/ratings/test.csv')
movies = pd.read_csv(
    '../../../rllm/datasets/rel-movielens1m/regression/movies.csv')

init= 'Given a user\'s past movie ratings in the format: Title, Genres, Rating (Note: Ratings range from 1 to 5)'
#end = 'What\'s the rating that the user will give to the movie(s)? Give a single number as rating if there\'s only one movie, else return like this: rating_for_movie1|rating_for_movie_2|...|rating_for_movie_n. Do not say anything else.'
end = 'What\'s the rating that the user will give to the movie? Give a single number as rating.'
train_prompts = df2prompts(train, init, end,sample_users=1000)
val_prompts = df2prompts(val, init, end,sample_users=None)
test_prompts = df2prompts(test, init, end,sample_users=None)
# test_prompts = df2prompts(test, init, end,sample_users=None)


write_jsonl('\n'.join(train_prompts),'train.json')
write_jsonl('\n'.join(val_prompts),'val.json')
write_jsonl('\n'.join(test_prompts),'test.json')

y_val = val['Rating']
y_test = test['Rating']






# gpt = GPTJ.LoRaQGPTJ(adapter=True, device=device,model_name='hivemind/gpt-j-6B-8bit')
gpt = GPTJ.LoRaQGPTJ(adapter=True, device=device)
train_configs={'learning_rate': 1e-5, 'batch_size': 1, 'epochs':1,  'weight_decay': 0.01, 'warmup_steps': 6}
gpt.finetune('data/train.json', 'data/val.json', train_configs, saving_checkpoint=False)

test_prompts = extract_prompts('data/test.json')
test_completions = extract_completion('data/test.json')

pred= query(gpt, test_prompts,bs=8)
print(pred)

pred_l=[]
for x in pred:
    try: 
        pred_l.append(int(x))
    except:
        pred_l.append(0)

y_pred = pd.DataFrame({'Rating':pred_l})['Rating']

y_test = pd.DataFrame({'Rating':[int(x[0]) for x in test_completions]})['Rating']


# acc = get_accuracy(y_pred, y_test)
# print(acc)

mae_loss = mae(y_test, y_pred)

time_end = time.time()

print(f"mae_loss: {mae_loss}")

print(f"Total time: {time_end - time_start}s")
# print(f"Total USD$: {total_cost}")