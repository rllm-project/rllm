import sys
sys.path.append('./')
sys.path.append('./../')
from re import A
import pandas as pd
import numpy as np
import openai
import argparse
import random
import time, json, os
from sklearn.utils import shuffle
from tqdm import tqdm
from utils.helper import log
from models.GPT3FineTuner import GPT3FineTuner

def get_in_context_prefix_prompts(jsonl_file,num_prompts):
    prefix_prompts = ''
    count = 0
    with open(jsonl_file) as fp:
        for i,line in enumerate(fp):
            count += 1
            if i >= num_prompts -1:
                break
            json_obj = json.loads(line)
            prefix_prompts += json_obj['prompt']
            prefix_prompts += json_obj['completion']
            prefix_prompts += ''
    print("Number of prefix prompts:",count)
    return prefix_prompts
    
def extract_completion(jsonl_file):
    completions = []
    with open(jsonl_file) as fp:
        for line in fp:
            json_obj = json.loads(line)
            completions.append(json_obj['completion'])
    return completions

def extract_subset(jsonl_file,num_prompts,random_state):
    training_set = []
    with open(jsonl_file) as fp:
        for j,line in enumerate(fp):
            training_set.append(line)
    training_set = pd.DataFrame(training_set)
    subfname = jsonl_file.split('.jsonl')[0]+"_subset.jsonl"
    shuffled = shuffle(training_set,random_state=random_state)
    shuffled = shuffled.values.tolist()
    subset = shuffled[:num_prompts]
    jsonl = ''.join(subset)
    
    with open(subfname, 'w') as f:
        f.write(jsonl)
    return subfname


def extract_random_incontext_prompts(jsonl_file:list,num_prompts,target_jsonl,random_state):
    prefix_prompts = []
    for jf in jsonl_file:
        with open(jf) as fp:
            for i,line in enumerate(fp):
                json_obj = json.loads(line)
                prefix_prompts.append(json_obj['prompt']+json_obj['completion']+' ')
    prefix_prompts = pd.DataFrame(prefix_prompts)
    in_context_prompts = []
    with open(target_jsonl) as fp:
        for line in fp:
            shuffled = shuffle(prefix_prompts,random_state=random_state)
            json_obj = json.loads(line)
            prefix_prompt = ''
            for j in range(num_prompts):
                prefix_prompt+=shuffled.values[j][0]
            in_context_prompts.append(prefix_prompt+json_obj['prompt'])
    # print(len(in_context_prompts),in_context_prompts[0])
    return in_context_prompts

def load_subset(jsonl_file:list,num_prompts,target_jsonl,random_state):
    prefix_prompts = []
    for jf in jsonl_file:
        with open(jf) as fp:
            for i,line in enumerate(fp):
                json_obj = json.loads(line)
                prefix_prompts.append(json_obj['prompt']+json_obj['completion']+' ')
    prefix_prompts = pd.DataFrame(prefix_prompts)
    in_context_prompts = []
    with open(target_jsonl) as fp:
        for line in fp:
            shuffled = shuffle(prefix_prompts,random_state=random_state)
            json_obj = json.loads(line)
            prefix_prompt = ''
            for j in range(num_prompts):
                prefix_prompt+=shuffled.values[j][0]
            in_context_prompts.append(prefix_prompt+json_obj['prompt'])
    # print(len(in_context_prompts),in_context_prompts[0])
    return in_context_prompts

def prompt2value(x):
    # print("Output:",x)
    c = x.strip().split('@@@')[0]
    if c == '':
        return None
    try:
        return int(c)
    except:
        return c

def load_jsonl(did, idx, context=False,feature_name=False):
    jsonl_files = {}
    if feature_name:
        for mode in ['train', 'val', 'test']:
            jsonl_files[mode] = f'data/{did}_split{idx}_{mode}_context_{context}_feature_names.jsonl'
    else:
        for mode in ['train', 'val', 'test']:
            jsonl_files[mode] = f'data/{did}_split{idx}_{mode}_context_{context}.jsonl'
    return jsonl_files

def extract_prompts(jsonl_file,in_context_prefix=''):
    test_prompts = []
    with open(jsonl_file,encoding='utf-8') as fp:
        for line in fp:
            json_obj = json.loads(line)
            test_prompts.append(in_context_prefix+json_obj['prompt'])
    return test_prompts

def generate(gpt, text_lst, max_token=10, batch_size=2):
    gpt.model.eval()
    outputs = []
    for i in tqdm(np.arange(0, len(text_lst), batch_size)):
        texts = text_lst[i:min(i + batch_size, len(text_lst))]
        prompt = gpt.tokenizer(texts, truncation=True, padding = True, max_length=1024, return_tensors='pt')
        prompt = {key: value.to(gpt.device) for key, value in prompt.items()}
        outs = gpt.model.generate(**prompt, max_new_tokens=max_token, pad_token_id=gpt.tokenizer.eos_token_id, do_sample=True, early_stopping = True)
        outs = gpt.tokenizer.batch_decode(outs, skip_special_tokens=True)
        outputs += outs
    return outputs

def generate_output(gpt3_fine_tuner, val_prompts):
    ans, bs, count = [], 20, 0
    while count < len(val_prompts):
        start, end = count, min(count + bs, len(val_prompts))
        count = end
        batch = val_prompts[start:end]
        # print('Input:',batch)
        ans += gpt3_fine_tuner.query(batch)
    return [prompt2value(x) for x in ans]

def generate_output_in_context(prompts, use_model):
    ans, bs, count = [], 20, 0
    while count < len(prompts):
        start, end = count, min(count + bs, len(prompts))
        count = end
        batch = prompts[start:end]
        outputs = openai.Completion.create(model=use_model,prompt = batch, temperature=0)
        ans += [outputs['choices'][i]['text'] for i in range(len(batch))]
    return [prompt2value(x) for x in ans]

def get_accuracy(y_pred_val, y_val):
    acc_val = (y_pred_val == y_val).mean()
    acc_val = round(acc_val * 100, 2)
    return acc_val

def query(gpt, prompts, bs=10):
    outputs = generate(gpt, prompts, batch_size=bs)
    ans = []
    for txt in outputs:
        try:
            output = prompt2value(txt.split('@@@')[0].split('###')[-1])
        except:
            output = txt
        ans.append(output)
    return ans