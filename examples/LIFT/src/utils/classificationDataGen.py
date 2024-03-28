import sys
sys.path.append('./')
sys.path.append('./../')
import pandas as pd
import numpy as np
import os, random, itertools
from utils import mnist
# from utils import *
import sklearn.datasets as datasets
from functools import partial

class dataGenerate(object):
    """
    A class of functions for generating jsonl datasets for classification tasks.
    """
    def __init__(self, seed = 123):
        self.seed = 123
        
    def data2text(self, row, integer = False, label = True, 
                  context = False, feature_names = None, target_names = None, init = '', end = ''):
        if context:
            prompt = init
            for i in range(len(row)-label):
                prompt += "%s=%.4f, " % (feature_names[i], row[i])
            prompt += end

            if not label:
                return prompt
            else:
                completion = "%s" % target_names[int(row['y'])]
                return "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)
        else:
            prompt = "When we have " 
            for i in range(len(row)-label):
                if integer:
                    prompt += "x%d=%d, " % (i+1, row[i])
                else:
                    prompt += "x%d=%.4f, " % (i+1, row[i]) 
            prompt += "what should be the y value?"
            
            if not label:
                return prompt
            else:
                completion = "%d" % row['y']
                return "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)
    
    def df2jsonl(self, df, filename, integer = False, 
                 context = False, feature_names = None, target_names = None, init = '', end = ''):
        jsonl = '\n'.join(df.apply(func = partial(self.data2text, 
                                                  integer = integer, 
                                                  context = context, 
                                                  feature_names = feature_names, 
                                                  target_names = target_names, 
                                                  init = init, 
                                                  end = end), axis = 1).tolist())
        fpath = os.path.join('data', filename)
        with open(fpath, 'w') as f:
            f.write(jsonl)
        return fpath
            
    def array2prompts(self, X, integer = False,
                     context = False, feature_names = None, target_names = None, init = '', end = ''):
        return list(map(partial(self.data2text, 
                                integer = integer, 
                                label = False,
                                context = context, 
                                feature_names = feature_names, 
                                target_names = target_names, 
                                init = init, 
                                end = end
                               ), X))
    
    def data_split(self, X, y):
        n = X.shape[0]
        idx = np.arange(n)
        random.shuffle(idx)
        train_idx, valid_idx, test_idx = idx[:int(.6*n)], idx[int(.6*n):int(.8*n)], idx[int(.8*n):]
        X_train, X_valid, X_test = X[train_idx], X[valid_idx], X[test_idx]
        y_train, y_valid, y_test = y[train_idx], y[valid_idx], y[test_idx]
        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def gridX_generate(self, X, resolution = 50):
        # lb = np.apply_along_axis(min, 0, X)
        # ub = np.apply_along_axis(max, 0, X)
        
        # lb, ub = lb - rang*0.2, ub + rang*0.2

        # X_grid = np.linspace(lb, ub, resolution).T
        # X_grid = np.array(list(itertools.product(*X_grid)))

        # h = 0.02
        lb = np.min(X, axis=0)[0]
        ub = np.max(X, axis=0)[0]
        rang = ub - lb
        h = rang/resolution
        xx, yy = np.meshgrid(np.arange(lb, ub, h),
                            np.arange(lb, ub, h))
        X_grid = np.c_[xx.ravel(), yy.ravel()]

        grid_prompts = self.array2prompts(X_grid)
        return pd.DataFrame(X_grid), grid_prompts
    
    def blobs(self, n, p, num_class, resolution = 30, outliers=None):
        X, y = datasets.make_blobs(n_samples = n, centers = num_class, n_features = p, random_state = self.seed)
        X_train, X_valid, X_test, y_train, y_valid, y_test = self.data_split(X, y)
        
        
        if outliers is not None:
            X_out, y_out = outliers
            X_train = np.concatenate([X_train, X_out], axis = 0)
            y_train = np.concatenate([y_train, y_out], axis = 0)
        
        train_df, valid_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)
        train_df['y'], valid_df['y'], test_df['y'] = y_train, y_valid, y_test
        
        self.train_jsonl = self.df2jsonl(train_df, 'blobs_n_%d_p_%d_class_%d_train.jsonl'%(n,p,num_class))
        self.val_jsonl = self.df2jsonl(valid_df, 'blobs_n_%d_p_%d_class_%d_valid.jsonl'%(n,p,num_class))
        
        test_prompts = self.array2prompts(X_test)
        
        if p == 2:
            grid_df, grid_prompts = self.gridX_generate(X, resolution = resolution)
            return train_df, valid_df, test_df, test_prompts, grid_df, grid_prompts
        else:
            return train_df, valid_df, test_df, test_prompts
        
    def moons(self, n, noise = 0.1, resolution = 30):
        """
        n: number of samples
        noise: Standard deviation of Gaussian noise added to the data
        resolution: resolution of the grid set
        """
        X, y = datasets.make_moons(n_samples = n, noise = noise, random_state = self.seed)
        X_train, X_valid, X_test, y_train, y_valid, y_test = self.data_split(X, y)
        train_df, valid_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)
        train_df['y'], valid_df['y'], test_df['y'] = y_train, y_valid, y_test
        
        self.train_jsonl = self.df2jsonl(train_df, 'moons_n_%d_noise_%.2f_train.jsonl'%(n,noise))
        self.val_jsonl = self.df2jsonl(valid_df, 'moons_n_%d_noise_%.2f_valid.jsonl'%(n,noise))
        
        test_prompts = self.array2prompts(X_test)
        
        grid_df, grid_prompts = self.gridX_generate(X, resolution = resolution)
        return train_df, valid_df, test_df, test_prompts, grid_df, grid_prompts
    
    def rolls(self, n, noise = 0.1, resolution = 30):
        """
        n: number of samples
        noise: Standard deviation of Gaussian noise added to the data
        resolution: resolution of the grid set
        """
        X, y = datasets.make_swiss_roll(n_samples = n, noise = noise, random_state = self.seed)
        X_train, X_valid, X_test, y_train, y_valid, y_test = self.data_split(X, y)
        train_df, valid_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)
        train_df['y'], valid_df['y'], test_df['y'] = y_train, y_valid, y_test
        
        self.train_jsonl = self.df2jsonl(train_df, 'moons_n_%d_noise_%.2f_train.jsonl'%(n,noise))
        self.val_jsonl = self.df2jsonl(valid_df, 'moons_n_%d_noise_%.2f_valid.jsonl'%(n,noise))
        
        test_prompts = self.array2prompts(X_test)
        
        grid_df, grid_prompts = self.gridX_generate(X, resolution = resolution)
        return train_df, valid_df, test_df, test_prompts, grid_df, grid_prompts
    
    def circles(self, n, noise = 0.1, factor = 0.8, resolution = 30):
        X, y = datasets.make_circles(n_samples = n, noise = noise, random_state = self.seed, factor = factor)
        X_train, X_valid, X_test, y_train, y_valid, y_test = self.data_split(X, y)
        train_df, valid_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)
        train_df['y'], valid_df['y'], test_df['y'] = y_train, y_valid, y_test
        
        self.train_jsonl = self.df2jsonl(train_df, 'circles_n_%d_noise_%.2f_factor_%.1f_train.jsonl'%(n,noise,factor))
        self.val_jsonl = self.df2jsonl(valid_df, 'circles_n_%d_noise_%.2f_factor_%.1f_valid.jsonl'%(n,noise,factor))
        
        test_prompts = self.array2prompts(X_test)
        
        grid_df, grid_prompts = self.gridX_generate(X, resolution = resolution)
        return train_df, valid_df, test_df, test_prompts, grid_df, grid_prompts   
    
    def gaussian9cluster(self, n, noise = 1, resolution = 30):
        X, y = [], []
        label = 0
        for i in [-10, 0, 10]:
            for j in [-10, 0, 10]:
                label += 1
                mean = [i, j]
                cov = noise * np.diag(np.ones(2))
                X.append(np.random.multivariate_normal(mean, cov, n//9))
                y.append(np.ones(n//9) * label)
        X, y = np.concatenate(X), np.concatenate(y)
        X_train, X_valid, X_test, y_train, y_valid, y_test = self.data_split(X, y)
        train_df, valid_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)
        train_df['y'], valid_df['y'], test_df['y'] = y_train, y_valid, y_test       
        
        self.train_jsonl = self.df2jsonl(train_df, 'nineclusters_n_%d_noise_%.2f_train.jsonl'%(n,noise))
        self.val_jsonl = self.df2jsonl(valid_df, 'nineclusters_n_%d_noise_%.2f_valid.jsonl'%(n,noise))
        
        test_prompts = self.array2prompts(X_test)
        
        grid_df, grid_prompts = self.gridX_generate(X, resolution = resolution)
        return train_df, valid_df, test_df, test_prompts, grid_df, grid_prompts  
    
    def twoCircles(self, n, noise = 0.05, factor = 0.5, resolution = 30):
        X1, y1 = datasets.make_circles(n_samples = n//2, noise = noise, random_state = self.seed, factor = factor)
        X2, y2 = datasets.make_circles(n_samples = n//2, noise = noise, random_state = self.seed * 2, factor = factor)
        X2.T[0] += 3
        X, y = np.concatenate([X1, X2]), np.concatenate([y1, 1-y2])
        
        X_train, X_valid, X_test, y_train, y_valid, y_test = self.data_split(X, y)
        train_df, valid_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)
        train_df['y'], valid_df['y'], test_df['y'] = y_train, y_valid, y_test
        
        self.train_jsonl = self.df2jsonl(train_df, 'twoCircles_n_%d_noise_%.2f_factor_%.1f_train.jsonl'%(n,noise,factor))
        self.val_jsonl = self.df2jsonl(valid_df, 'twoCircles_n_%d_noise_%.2f_factor_%.1f_valid.jsonl'%(n,noise,factor))
        
        test_prompts = self.array2prompts(X_test)
        
        grid_df, grid_prompts = self.gridX_generate(X, resolution = resolution)
        return train_df, valid_df, test_df, test_prompts, grid_df, grid_prompts 
        
    def mnist(self):
        random.seed(self.seed)
        
        X, y, X_test, y_test = mnist.load()
        idx = np.arange(60000)
        random.shuffle(idx)
        train_idx, valid_idx = idx[:50000], idx[50000:]
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        train_df, valid_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)
        train_df['y'], valid_df['y'], test_df['y'] = y_train, y_valid, y_test
        
        self.train_jsonl = self.df2jsonl(train_df, 'mnist_train.jsonl')
        self.val_jsonl = self.df2jsonl(valid_df, 'mnist_valid.jsonl')
        
        test_prompts = self.array2prompts(X_test)
        return train_df, valid_df, test_df, test_prompts
    
    def permuted_mnist(self):
        random.seed(self.seed)
        
        X, y, X_test, y_test = mnist.load()
        permutation = np.arange(X.shape[1])
        random.shuffle(permutation)
        X0 = np.apply_along_axis(lambda x: x[permutation], 1, X)
        X_test = np.apply_along_axis(lambda x: x[permutation], 1, X_test)
        
        idx = np.arange(60000)
        random.shuffle(idx)
        train_idx, valid_idx = idx[:50000], idx[50000:]
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        train_df, valid_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)
        train_df['y'], valid_df['y'], test_df['y'] = y_train, y_valid, y_test
        
        self.train_jsonl = self.df2jsonl(train_df, 'permuted_mnist_train.jsonl')
        self.val_jsonl = self.df2jsonl(valid_df, 'permuted_mnist_valid.jsonl')
        
        test_prompts = self.array2prompts(X_test)
        return train_df, valid_df, test_df, test_prompts
    
    def iris(self, context = True):
        data = datasets.load_iris()
        X, y = data['data'], data['target']

        X_train, X_valid, X_test, y_train, y_valid, y_test = self.data_split(X, y)
        train_df, valid_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)
        train_df['y'], valid_df['y'], test_df['y'] = y_train, y_valid, y_test       
        
        self.train_jsonl = self.df2jsonl(train_df, 'iris_context_%s_train.jsonl' % context,
                      context = context, 
                      feature_names = data.feature_names, 
                      target_names = data.target_names, 
                      init = 'Given a iris plant with ', 
                      end = 'what is the type of it?')
        self.val_jsonl = self.df2jsonl(valid_df, 'iris_context_%s_valid.jsonl' % context,
                      context = context, 
                      feature_names = data.feature_names,
                      target_names = data.target_names, 
                      init = 'Given a iris plant with ', 
                      end = 'what is the type of it?')
        
        test_prompts = self.array2prompts(X_test,
                                          context = context, 
                                          feature_names = data.feature_names,
                                          target_names = data.target_names, 
                                          init = 'Given a iris plant with ', 
                                          end = 'what is the type of it?')
        return train_df, valid_df, test_df, test_prompts
        
    def breastCancer(self, context = True):
        data = datasets.load_breast_cancer()
        X, y = data['data'], data['target']

        X_train, X_valid, X_test, y_train, y_valid, y_test = self.data_split(X, y)
        train_df, valid_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)
        train_df['y'], valid_df['y'], test_df['y'] = y_train, y_valid, y_test       
        
        self.train_jsonl = self.df2jsonl(train_df, 'breastCancer_context_%s_train.jsonl' % context,
                      context = context, 
                      feature_names = data.feature_names, 
                      target_names = data.target_names, 
                      init = 'Given a cell nuclei with ', 
                      end = 'what is the condition?')
        self.val_jsonl = self.df2jsonl(valid_df, 'breastCancer_context_%s_valid.jsonl' % context,
                      context = context, 
                      feature_names = data.feature_names,
                      target_names = data.target_names, 
                      init = 'Given a cell nuclei with ', 
                      end = 'what is the condition?')
        
        test_prompts = self.array2prompts(X_test,
                                          context = context, 
                                          feature_names = data.feature_names,
                                          target_names = data.target_names, 
                                          init = 'Given a cell nuclei with ', 
                                          end = 'what is the condition?')
        return train_df, valid_df, test_df, test_prompts
    
    def wine(self, context = True):
        data = datasets.load_wine()
        X, y = data['data'], data['target']

        X_train, X_valid, X_test, y_train, y_valid, y_test = self.data_split(X, y)
        train_df, valid_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)
        train_df['y'], valid_df['y'], test_df['y'] = y_train, y_valid, y_test       
        
        self.train_jsonl = self.df2jsonl(train_df, 'wine_context_%s_train.jsonl' % context,
                      context = context, 
                      feature_names = data.feature_names,
                      target_names = data.target_names, 
                      init = 'Given the wine with', 
                      end = 'which class does it belong to?')
        self.val_jsonl = self.train_jsonl = self.df2jsonl(train_df, 'wine_context_%s_valid.jsonl' % context,
                      context = context, 
                      feature_names = data.feature_names,
                      target_names = data.target_names, 
                      init = 'Given the wine with ', 
                      end = 'which class does it belong to?')
        
        test_prompts = self.array2prompts(X_test,
                                          context = context, 
                                          feature_names = data.feature_names,
                                          target_names = data.target_names, 
                                          init = 'Given the wine with ', 
                                          end = 'which class does it belong to?')
        return train_df, valid_df, test_df, test_prompts
    
    def meshgrid_prompt(self, X, resolution = 50):
        x_min, x_max =  -6, 6
        y_min, y_max = -6, 6
        h = 0.02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = nnet2(np.c_[xx.ravel(), yy.ravel()])
        
        lb = np.apply_along_axis(min, 0, X)
        ub = np.apply_along_axis(max, 0, X)
        rang = ub - lb
        lb, ub = lb - rang*0.2, ub + rang*0.2

        X_grid = np.linspace(lb, ub, resolution).T
        X_grid = np.array(list(itertools.product(*X_grid)))

        grid_prompts = self.array2prompts(X_grid)
        return pd.DataFrame(X_grid), grid_prompts
    
    def neural_net(self, labeling_func, n, name='nnet', ranges=(-6, 6), noise=0, resolution=30, corrupted =0.05):
        lb, ub = ranges
        xy_min = [lb, lb]
        xy_max = [ub, ub]
        X = np.random.uniform(low=xy_min, high=xy_max, size=(n,2))
        y = labeling_func(X)
        
        X_train, X_valid, X_test, y_train, y_valid, y_test = self.data_split(X, y)
        if corrupted > 0:
            # corrupt here
            n = len(y_train)
            m = int(n * corrupted)
            inds = random.sample(range(1, n), m)
            for i in inds:
                y_train[i] = 1 - y_train[i]

        train_df, valid_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)
        train_df['y'], valid_df['y'], test_df['y'] = y_train, y_valid, y_test       
        
        self.train_jsonl = self.df2jsonl(train_df, 'nnet_%s_n_%d_corrupted_%.2f_train.jsonl'%(name, n,corrupted))
        self.val_jsonl = self.df2jsonl(valid_df, 'nnet_%s_n_%d_corrupted_%.2f_valid.jsonl'%(name, n,corrupted))
        
        test_prompts = self.array2prompts(X_test)
        
        grid_df, grid_prompts = self.gridX_generate(X, resolution = resolution)
        return train_df, valid_df, test_df, test_prompts, grid_df, grid_prompts 



