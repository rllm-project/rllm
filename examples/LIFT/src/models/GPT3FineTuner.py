import openai
import time
import matplotlib.pyplot as plt
import numpy as np

def L2error(y1, y2):
    try:
        return np.linalg.norm(y1.reshape(-1) - y2.reshape(-1))
    except AttributeError:
        try:
            return np.linalg.norm(y1 - y2.reshape(-1))
        except AttributeError:
            try:
                return np.linalg.norm(y1.reshape(-1) - y2)
            except AttributeError:
                return np.linalg.norm(y1 - y2)

def RMSE(a,b):
    a = np.array(a)
    b = np.array(b)
    if a.shape != b.shape:
        raise ValueError('RMSE input error')
    return np.mean((a-b)**2)**0.5


def RMSE_woo(a,b,threshold=20):
    a = np.array(a)
    b = np.array(b)
    if a.shape != b.shape:
        raise ValueError('RMSE input error')
    std = RMSE(a,b)
    outlier_flag = (np.abs(a-b) > std*threshold)
    num_outlier = np.sum(outlier_flag)
    
    return RMSE(a[~outlier_flag],b[~outlier_flag]), num_outlier

class GPT3FineTuner(object):
    def __init__(self,config:dict,train_jsonl,valid_jsonl):
        self.config = config
        self.train_jsonl=train_jsonl
        self.valid_jsonl=valid_jsonl
        
        self.file_info = openai.File.create(file = open(train_jsonl), purpose = 'fine-tune')
        self.training_file_id   = self.file_info['id']
        self.file_info = openai.File.create(file = open(valid_jsonl), purpose = 'fine-tune')
        self.validation_file_id   = self.file_info['id']

    
    def init_model(self, clf_cfgs):
        print("Initialize a new GPT3 Model")
        self.fine_tuned = False
        if clf_cfgs['n_classes'] == 0:
            self.ft_info = openai.FineTune.create(training_file = self.training_file_id,
                                    validation_file = self.validation_file_id,
                                    model = self.config['model_type'],
                                    n_epochs = self.config['num_epochs'],
                                    batch_size = self.config['batch_size'],
                                    # learning_rate_multiplier = self.config['learning_rate_multiplier'],
                                    #prompt_loss_weight = prompt_loss_weight,
                                    #compute_classification_metrics = compute_classification_metrics,
                                    #classification_n_classes = classification_n_classes,
                                    #classification_positive_class = classification_positive_class,
                                    #classification_betas = classification_betas
                                    )
        elif clf_cfgs['n_classes'] == 2:
            self.ft_info = openai.FineTune.create(training_file = self.training_file_id,
                                    validation_file = self.validation_file_id,
                                    model = self.config['model_type'],
                                    n_epochs = self.config['num_epochs'],
                                    batch_size = self.config['batch_size'],
                                    # learning_rate_multiplier = self.config['learning_rate_multiplier'],
                                    #prompt_loss_weight = prompt_loss_weight,
                                    compute_classification_metrics = True,
                                    classification_n_classes = clf_cfgs['n_classes'],
                                    classification_positive_class = clf_cfgs['positive_class'],
                                    #classification_betas = classification_betas
                                    )
        elif clf_cfgs['n_classes'] > 2:
            self.ft_info = openai.FineTune.create(training_file = self.training_file_id,
                                    validation_file = self.validation_file_id,
                                    model = self.config['model_type'],
                                    n_epochs = self.config['num_epochs'],
                                    batch_size = self.config['batch_size'],
                                    # learning_rate_multiplier = self.config['learning_rate_multiplier'],
                                    #prompt_loss_weight = prompt_loss_weight,
                                    compute_classification_metrics = True,
                                    classification_n_classes = clf_cfgs['n_classes'],
                                    #classification_positive_class = classification_positive_class,
                                    #classification_betas = classification_betas
                                    )
                                
        self.ft_id = self.ft_info['id']

    def fine_tune(self, clf_cfgs={'n_classes': 0, 'positive_class': None}):
        self.init_model(clf_cfgs)
        self.finetune_status = None
        while(self.finetune_status != 'succeeded'):
            self.ft_info = openai.FineTune.retrieve(id=self.ft_id)
                
            time.sleep(10)
            if self.finetune_status != self.ft_info['status']:
                self.finetune_status = self.ft_info['status']
                print(self.finetune_status)
        self.ft_model = self.ft_info['fine_tuned_model']
        print('fine-tune id: ',self.ft_id)
        print('fine-tune model: ',self.ft_info['fine_tuned_model'])

    
    def query(self, prompts):
        flag = True
        while(flag):
            try:
                outputs =  openai.Completion.create(model = self.ft_model,prompt = prompts, temperature=0)
                flag = False
            except Exception as e: 
                print(e)
                print("Still Loading the model...")
                flag = True
                time.sleep(1)
        return [outputs['choices'][i]['text'] for i in range(len(prompts))]
        # try:
        #     return float(output.split('@@@')[0])
        # except:
        #     return None

    def eval(self,n_train,test_prompts,test_df,resolution,y_name='y',plot=False,X_grid=None,grid_prompts=None,y_grid=None,file_name=None):
        """
            number of valid samples
            L2 error on the valid samples
        """

        y_test_outputs = list(map(self.query,test_prompts))

        # print(y_test_outputs)

        # test_df["y_test_output"] = y_test_outputs

        
        valid_test_y = [test_df[y_name][i] for i in range(len(y_test_outputs)) if y_test_outputs[i] != None]
        valid_test_y_outputs = [y_test_outputs[i] for i in range(len(y_test_outputs)) if y_test_outputs[i] != None]

        # print(valid_test_y)

        
        print("Valid #outputs/Total #outputs:%d/%d" % (len(valid_test_y),len(y_test_outputs)))
        err_rate = np.mean(np.where(np.sign(valid_test_y_outputs)==valid_test_y,0,1))
        print('Error Rate     : %.4f' % err_rate)

        if plot and X_grid is not None and grid_prompts is not None:
            
            y_grid_outputs = list(map(self.query,grid_prompts))
        else:
            y_grid_outputs = None
        
        return y_test_outputs,y_grid_outputs,len(valid_test_y), err_rate

