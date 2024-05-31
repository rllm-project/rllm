import numpy as np
from sklearn import metrics
from numpy.typing import NDArray
import tiktoken


### metrics for regression

def mae(truth: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    return metrics.mean_absolute_error(truth, pred)


def mse(truth: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    return metrics.mean_squared_error(truth, pred)


def rmse(truth: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    return metrics.mean_squared_error(truth, pred, squared=False)


def r2(truth: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    return metrics.r2_score(truth, pred)


### metrics for classification

# metrics for binary/multiclass classification
def accuracy(truth: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    if pred.ndim == 1:
        label = pred > 0.5
    else:
        label = pred.argmax(axis=1)
    return metrics.accuracy_score(truth, label)


def log_loss(truth: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    if pred.ndim == 1 or pred.shape[1] == 1:
        prob = np.sigmoid(pred)
    else:
        prob = np.softmax(pred, axis=1)
    return metrics.log_loss(truth, prob)


# metrics only for multilabel classification
def macro_f1_score(truth: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    assert pred.ndim > 1
    label = np.where(pred > 0.5, 1, 0)
    return metrics.f1_score(truth, label, average='macro')


def micro_f1_score(truth: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    assert pred.ndim > 1
    label = np.where(pred > 0.5, 1, 0)
    return metrics.f1_score(truth, label, average='micro')


### utils for cost


def get_llm_chat_cost(text: str, type: str) -> float:
    r"""
    Return LLM cost, accroding to OpenAI ChatGPT 3.5 turbo charging rules on 25/02/2024.
    More infomation: https://openai.com/pricing
    For input tokens, price is $0.0005 / 1K tokens.
    For output tokens, price is $0.0015 / 1K tokens."""
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(text))
    if type == 'input':
        return 0.0005 * num_tokens / 1000
    elif type == 'output':
        return 0.0015 * num_tokens / 1000


def get_llm_emb_cost(text: str) -> float:
    r"""
    Return LLM embedding cost, accroding to OpenAI ada v2 model charging rules on 25/02/2024.
    More infomation: https://openai.com/pricing
    Price is $0.00010 / 1K tokens"""
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(text))
    return 0.00010 * num_tokens / 1000

def get_lm_emb_cost(text: str) -> float:
    r"""
    Return LM embedding cost.
    Model all-MiniLM-L6-v2 (augmented BERT model. Paper: MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers) has 107M parameters,
    and we assume ada v2 is based on GPT3 with 175B parameters,
    so the parameters differ by about a factor of 1000 between the two models. 
    Let's assume the cost difference scales with the square of the parameter factor, 
    that is 10^6 times, namely $0.00000000010 / 1K token. 
    """
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(text))
    return 0.00000000010 * num_tokens / 1000