import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import get_linear_schedule_with_warmup, AdamW

from bitsandbytes.optim import Adam8bit
import random 

import os, time
from tqdm import tqdm
import json
import numpy as np
import gc 
gc.collect()

from models.lora_gptj_ops import GPTJForCausalLM, GPTJBlock, add_adapters




def setup_for_distributed_mode(model: nn.Module, optimizer: torch.optim.Optimizer, device: object, n_gpu: int = 1,
                               local_rank: int = -1,
                               fp16: bool = False,
                               fp16_opt_level: str = "O1"):
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)
    return model, optimizer


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class GPTJDataset(Dataset):
    def __init__(self, json_lst, tokenizer, max_length=1024):
        texts = []
        completion_lens = []
        for row in json_lst:
            t = ' '.join(row.values())
            texts.append(t)
            l = len(tokenizer.tokenize(row['completion']))
            completion_lens.append(l)
        
        tokens = tokenizer(texts, truncation=True, padding = True, max_length=max_length, return_tensors='pt')
        self.input_ids = tokens['input_ids']
        self.attention_mask = tokens['attention_mask']
        self.labels = []
        for i in range(len(self.input_ids)):
            b_labels = self.input_ids[i].clone()
            b_labels[:-completion_lens[i]] = -100
            self.labels.append(b_labels)
            
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx] 

class LoRaQGPTJ:
    def __init__(self, model_name='EleutherAI/gpt-j-6B', adapter=True, device=torch.device('cuda:0'), model_path='../results/gpt-j/') -> None:
        transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock  # monkey-patch GPT-J
        self.config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        # Define PAD Token = EOS Token = 50256 -- new modifications
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPTJForCausalLM.from_pretrained("hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)
        self.model.config.use_cache = False
        # finetune
        if adapter:
            add_adapters(self.model)
        if not(model_name == 'EleutherAI/gpt-j-6B'):
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            self.model = GPTJForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)

        self.device = device
        self.model = self.model.to(self.device)
        self.model_path = model_path
    
    def load_networks(self, model_name):
        self.model = GPTJForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).to(self.device)

    def prepare_data(self, jsonl_path):
        with open(jsonl_path, 'r') as json_file:
            json_lst = list(json_file)

        txt_list = []
        for json_str in json_lst:
            result = json.loads(json_str)
            txt_list.append(result)
        
        data = GPTJDataset(txt_list, self.tokenizer)
        
        return data

    def finetune(self, train_jsonl_path, val_jsonl_path, train_configs={'batch_size': 8, 'epochs': 20, 'learning_rate': 1e-3, 'weight_decay': 0.01, 'warmup_steps': 20}, saving_checkpoint=False):
        train_data = self.prepare_data(train_jsonl_path)
        val_data = self.prepare_data(val_jsonl_path)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data,
            num_replicas=4,
            rank=train_configs['local_rank']
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_data,
            num_replicas=4,
            rank=train_configs['local_rank']
        )
        data_loader = DataLoader(train_data, batch_size=train_configs['batch_size'], sampler=train_sampler)
        val_loader = DataLoader(val_data, batch_size=train_configs['batch_size'], sampler=val_sampler)
        total_steps = len(data_loader) * train_configs['epochs']

        # params 
        params_for_optimizer = []
        for name, param in self.model.named_parameters():
            if "adapter" in name: # "attn" in name and 
#                 print(f"Setting {name} requires_grad=True")
                param.requires_grad = True
                params_for_optimizer.append(param)
                # nn.init.zeros_(param)
                # nn.init.xavier_uniform_(param)
            else:
#                 print(f"Setting {name} requires_grad=False")
                param.requires_grad = False

        # self.model.gradient_checkpointing_enable()

        optimizer = Adam8bit(params_for_optimizer, lr=train_configs['learning_rate'], weight_decay=0.01) # freeze the W


        setup_for_distributed_mode(self.model, optimizer, device=self.device, n_gpu=torch.cuda.device_count(), local_rank=train_configs['local_rank'])


        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = train_configs['warmup_steps'], 
                                            num_training_steps = total_steps)


        best_loss = np.inf
        # with torch.cuda.amp.autocast():
        for epoch in range(train_configs['epochs']):
            # self.model.train()
            tqdm_object = tqdm(data_loader, total=len(data_loader), desc=f"Epoch: {epoch + 1}")
            loss_meter = AverageMeter()
            for batch in tqdm_object:
                self.model.zero_grad()
                outputs = self.model(batch[0].to(self.device),
                    labels=batch[2].to(self.device), 
                    attention_mask = batch[1].to(self.device),
                    token_type_ids=None
                )
                loss = outputs[0]
                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_meter.update(loss.detach().item(), batch[0].shape[0])
                tqdm_object.set_postfix(train_loss=loss_meter.avg)
                # torch.cuda.empty_cache()
            
            val_loss = self.validate(val_loader)
#             print('Valilation loss: {:.4f}'.format(val_loss))
            if saving_checkpoint and val_loss < best_loss:
                print('Saving the best model with loss {:.4f}'.format(val_loss))
                best_loss = val_loss
                self.save_networks(self.model_path)

                
        

    def validate(self, val_loader):
        # ========================================
        #               Validation
        # ========================================
        self.model.eval()
        # Evaluate data for one epoch
        loss_meter = AverageMeter()
        tqdm_object = tqdm(val_loader, total=len(val_loader), desc='Validation')
        for batch in tqdm_object:   
            with torch.no_grad():
                outputs = self.model(batch[0].to(self.device),
                    labels=batch[2].to(self.device), 
                    attention_mask = batch[1].to(self.device),
                    token_type_ids=None
                )
                loss = outputs[0]  

            loss_meter.update(loss.detach().item(), batch[0].shape[0])
            tqdm_object.set_postfix(val_loss=loss_meter.avg)
        
        return loss_meter.avg
    
    def generate(self, text_lst, deterministic=True, max_token=10, batch_size=10, temperature=1.0):
        self.model.eval()
        outputs = []
        for i in np.arange(0, len(text_lst), batch_size):
            texts = text_lst[i:min(i + batch_size, len(text_lst))]
            prompt = self.tokenizer(texts, truncation=True, padding = True, max_length=1024, return_tensors='pt')
            prompt = {key: value.to(self.device) for key, value in prompt.items()}
            outs = self.model.generate(**prompt, max_new_tokens=max_token, pad_token_id=self.tokenizer.eos_token_id, do_sample=True, early_stopping = True, temperature=temperature)
            outs = self.tokenizer.batch_decode(outs, skip_special_tokens=True)
            outputs += outs
        return outputs


    def save_networks(self, output_dir = '../results/gpt-j/'):
        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model to %s" % output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


def test(texts, previous_token, end_token):
    y = [txt.split(end_token)[0].split(previous_token)[-1] for txt in texts]
    return y

# if __name__ == '__main__':
#     device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
#     gpt = LoRaQGPTJ(adapter=True, device=device)
#     train_jsonl = f"../datasets/test/compas_train.jsonl"
#     val_jsonl = f"../datasets/test/compas_test.jsonl"
#     test_jsonl = f"../datasets/test/compas_test.jsonl"

#     train_configs={'batch_size': 4, 'epochs': 10, 'learning_rate': 1e-4, 'weight_decay': 0.01, 'warmup_steps': 6}

#     gpt.finetune(train_jsonl, val_jsonl, train_configs)
    
#     texts = "The defendant, a 69-year-old male, was arrested for a felony. The specific charge is Aggravated Assault w/Firearm. The defendant has committed 0 juvenile misdemeanors, 0 juvenile felonies, 0 other juvenile delinquencies, and 0 prior convictions for other offenses. Will this defendant reoffend in two years? ###"
#     output = gpt.generate(texts)
#     print(output)
