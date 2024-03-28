import numpy as np
from functools import partial
import os
import json

def generate(gpt, text_lst, deterministic=True, max_token=10, batch_size=2):
    gpt.model.eval()
    outputs = []
    for i in np.arange(0, len(text_lst), batch_size):
        texts = text_lst[i:min(i + batch_size, len(text_lst))]
        prompt = gpt.tokenizer(texts, truncation=True, padding = True, max_length=1024, return_tensors='pt')
        prompt = {key: value.to(gpt.device) for key, value in prompt.items()}
        outs = gpt.model.generate(**prompt, max_new_tokens=max_token, pad_token_id=gpt.tokenizer.eos_token_id, do_sample=True, early_stopping = True)
        outs = gpt.tokenizer.batch_decode(outs, skip_special_tokens=True)
        outputs += outs
    return outputs


def query(gpt, prompts, bs=10):
    outputs = generate(gpt, prompts, batch_size=bs)
    ans = [txt.split('@@@')[0].split('###')[-1] for txt in outputs]
    return ans, outputs


def L2error(targets, predictions):
    return np.sqrt(np.mean((predictions.reshape(-1) - targets.reshape(-1) )**2))


def add_ending_symbol(prompts):
    ending_symbol = '###'
    if ending_symbol not in prompts[0]:
        return [p + ending_symbol for p in prompts]


def log(logf, msg, console_print=True):
    logf.write(msg + '\n')
    if console_print:
        print(msg)

# ####### ===== Prompt ========= ######

def data2text(row, label = True, init = '', end = ''):
    prompt = init 
    for i in range(len(row)-label):
        v = row[i]
        prompt += "%d " % (v)
    prompt += end

    if not label:
        final_prompt = f"{prompt}###"
    else:
        completion = "%d" %row['y']
        final_prompt = "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)
    return final_prompt

def df2jsonl(df, filename, init = '', end = ''):
    jsonl = '\n'.join(df.apply(func = partial(data2text, init = init, end = end), axis = 1).tolist())
    fpath = os.path.join('data', filename)
    with open(fpath, 'w') as f:
        f.write(jsonl)
    return fpath

def df2propmts(df, data2text_func, init = '', end = ''):
    jsonl = df.apply(func = partial(data2text_func, init = init, end = end), axis = 1).tolist()
    return jsonl

def write_jsonl(jsonl, filename):
    fpath = os.path.join('data', filename)
    with open(fpath, 'w',encoding='utf-8') as f:
        f.write(jsonl)
    return fpath

def array2prompts(X, init = '', end = ''):
    return list(map(partial(data2text, 
                            label = False,
                            init = init, 
                            end = end
                           ), X))
