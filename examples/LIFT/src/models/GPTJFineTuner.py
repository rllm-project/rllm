import sys
sys.path.append('./')
sys.path.append('./../')
import openai, os, time, torch, sys, importlib, json, copy
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import numpy as np
from models import lora_gptj
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from utils.helper import log

def get_accuracy(y_pred_val, y_val):
    acc_val = (np.array(y_pred_val) == np.array(y_val)).mean()
    acc_val = round(acc_val * 100, 2)
    return acc_val

class GPTJFineTuner(object):
    def __init__(self,config:dict,train_jsonl,valid_jsonl,cuda_idx = 0):
        self.config = config
        self.train_jsonl=train_jsonl
        self.valid_jsonl=valid_jsonl

        self.device = torch.device('cuda:%d' % cuda_idx) if torch.cuda.is_available() else 'cpu'
        torch.cuda.set_device(cuda_idx)
    
    def init_model(self):
        print('=====Initialize a new GPTJ Model=====')
        self.ft_model = lora_gptj.LoRaQGPTJ(adapter=True, device=self.device)

    def fine_tune(self):
        self.init_model()

    def generate(self, gpt, text_lst, max_token=10, batch_size=2):
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
        
    def prompt2value(self, x):
        # print("Output:",x)
        c = x.strip().split('@@@')[0]
        return float(c)
    
    def query(self, gpt, prompts, bs=10):
        outputs = self.generate(gpt, prompts, batch_size=bs)
        ans = []
        for txt in outputs:
            try:
                output = self.prompt2value(txt.split('@@@')[-1].split('###')[0])
            except:
                output = None
            ans.append(output)
        return ans


    def eval(self,valid_prompts,valid_df,test_prompts,test_df,logf,y_name='y',train_df = None,imbalance=False,flip=False):
        """
            number of valid samples
            L2 error on the valid samples
        """
        y_valid_outputs_, y_test_outputs_, len_valid_valid_y_, val_acc_list, test_acc_list = [], [], [], [], []
        best_idx = 0
        for model_idx in range(len(self.config['epochs'])):
            config = copy.deepcopy(self.config)
            epochs_ran = 0 if model_idx == 0 else self.config['epochs'][model_idx-1]
            config['epochs'] = self.config['epochs'][model_idx] - epochs_ran
            print('==== Epoch %.4f ====' % self.config['epochs'][model_idx])
            self.ft_model.finetune(self.train_jsonl, 
                self.valid_jsonl,
                config,
                saving_checkpoint = False)

            # validation
            y_valid_outputs = self.query(self.ft_model, valid_prompts, bs = 15)
            y_valid_outputs_.append(y_valid_outputs)
        
            valid_valid_y = [valid_df[y_name][i] for i in range(len(y_valid_outputs)) if y_valid_outputs[i] != None]
            valid_valid_y_outputs = [y_valid_outputs[i] for i in range(len(y_valid_outputs)) if y_valid_outputs[i] != None]

            len_valid_valid_y = len(valid_valid_y)
            print("| Valid Val #outputs/Total #outputs:%d/%d" % (len_valid_valid_y,len(y_valid_outputs)))
            len_valid_valid_y_.append(len_valid_valid_y)
            
            print(type(valid_valid_y_outputs), type(valid_valid_y))
            # from IPython import embed; embed()
            
            val_acc = get_accuracy(valid_valid_y_outputs, valid_valid_y)
            val_acc_list.append(val_acc)

            print('| Val Acc     : %.2f' % val_acc)
            if (val_acc < val_acc_list[best_idx]) or (np.isnan(val_acc_list[best_idx])):
                best_idx = model_idx
            
            # Testing
            y_test_outputs = self.query(self.ft_model, test_prompts, bs = 10)
            y_test_outputs_.append(y_test_outputs)

            valid_test_y = [test_df[y_name][i] for i in range(len(y_test_outputs)) if y_test_outputs[i] != None]
            valid_test_y_outputs = [y_test_outputs[i] for i in range(len(y_test_outputs)) if y_test_outputs[i] != None]
            print("Valid Test #outputs/Total #outputs:%d/%d" % (len(valid_test_y),len(y_test_outputs)))
            test_acc = get_accuracy(valid_test_y_outputs, valid_test_y)
            test_acc_list.append(test_acc)
            print('| Test Acc     : %.2f' % test_acc)
            
            if imbalance:
                if flip:
                    valid_valid_y = (-1*(valid_valid_y-1)).astype("int")
                    valid_valid_y_outputs = (-1*(valid_valid_y_outputs-1)).astype("int")
                    valid_test_y = (-1*(valid_test_y-1)).astype("int")
                    valid_test_y_outputs = (-1*(valid_test_y_outputs-1)).astype("int")

                precision_val = round(precision_score(valid_valid_y, valid_valid_y_outputs) * 100, 2)
                recall_val = round(recall_score(valid_valid_y, valid_valid_y_outputs) * 100, 2)
                f1_val = round(f1_score(valid_valid_y, valid_valid_y_outputs) * 100, 2)
                
                precision = round(precision_score(valid_test_y, valid_test_y_outputs) * 100, 2)
                recall = round(recall_score(valid_test_y, valid_test_y_outputs) * 100, 2)
                f1 = round(f1_score(valid_test_y, valid_test_y_outputs) * 100, 2)
                log(logf, f"val {self.config['epochs'][model_idx]} {val_acc} {f1_val} {precision_val} {recall_val}")
                log(logf, f"test {self.config['epochs'][model_idx]} {test_acc} {f1} {precision} {recall}")
            else:
                log(logf, f"{self.config['epochs'][model_idx]} {val_acc} {test_acc}")

        print('Selected epoch: %.4f' % self.config['epochs'][best_idx])
        self.best_idx = best_idx

        
        return y_test_outputs_[best_idx], len_valid_valid_y_,val_acc_list, test_acc_list

