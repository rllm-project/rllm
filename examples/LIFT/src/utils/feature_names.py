import os
import numpy as np
from functools import partial
import pandas as pd
import random

FEATURE_NAMES_54 = ['COMPACTNESS', 'CIRCULARITY', 'DISTANCE_CIRCULARITY', 'RADIUS_RATIO', 'PR.AXIS_ASPECT_RATIO', 'MAX.LENGTH_ASPECT_RATIO', 'SCATTER_RATIO', 'ELONGATEDNESS', 'PR.AXIS_RECTANGULARITY', 'MAX.LENGTH_RECTANGULARITY', 'SCALED_VARIANCE_MAJOR', 'SCALED_VARIANCE_MINOR', 'SCALED_RADIUS_OF_GYRATION', 'SKEWNESS_ABOUT_MAJOR', 'SKEWNESS_ABOUT_MINOR', 'KURTOSIS_ABOUT_MAJOR', 'KURTOSIS_ABOUT_MINOR', 'HOLLOWS_RATIO', 'Class']



def data2text_feature_name_48(row, integer = False, label = True):
    prompt = "Knowing a Teaching Assistant who's " 
    if row[0] == 1:
        prompt += 'an English speaker, '
    elif row[0] == 2:
        prompt += 'a non-English speaker, '
    prompt += "who teaches the course %d with instructor %d during " % (int(row[2]),row[1])
    if int(row[3]) == 1:
        prompt += "the summer session, "
    else:
        prompt += "the regular session, "
    prompt += 'with %d students, what is the rating for this Teaching Assistant?' % int(row[4])
    
    completion = "%d" % row['y']
    return "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)

# 'Whether_of_not_the_TA_is_a_native_English_speaker',1,2,'Summer_or_regular_semester','Class_size','Class_attribute'
def data2text_feature_name_23(row, integer = False, label = True):
    prompt = "In a family, the %d-year-old wife " % row[0]
    
    
    if int(row[4]) == 0:
        prompt += "does not believe in Islam. "
    elif int(row[4]) == 1:
        prompt += "believes in Islam. "
    else: 
        print("row[4]",row[4],type(row[4]))
        print(row[4] == 1)
        raise NotImplementedError
        
    if int(row[1]) == 1:
        prompt += 'She has a very low education level. '
    elif int(row[1]) == 2:
        prompt += 'She has a below-average education level. '
    elif int(row[1]) == 3:
        prompt += 'She has an above-average education level. '
    elif int(row[1]) == 4:
        prompt += 'She has a very high education level. '
    else: 
        raise NotImplementedError
        
    if int(row[5]) == 0:
        prompt += "She has a job. "
    elif int(row[5]) == 1:
        prompt += "She does not have a job. "
    else: 
        raise NotImplementedError
    
    if int(row[2]) == 1:
        prompt += 'The husband has a very low education level. '
    elif int(row[2]) == 2:
        prompt += 'The husband has a below-average education level. '
    elif int(row[2]) == 3:
        prompt += 'The husband has an above-average education level. '
    elif int(row[2]) == 4:
        prompt += 'The husband has a very high education level. '
    else: 
        raise NotImplementedError
    
    if int(row[6]) == 1:
        prompt += "He has a type A job. "
    elif int(row[6]) == 2:
            prompt += "He has a type B job. "
    elif int(row[6]) == 3:
        prompt += "He has a type C job. "
    elif int(row[6]) == 4:
        prompt += "He has a type D job. "
    else: 
        raise NotImplementedError
        
    prompt += "They've had %d children. " % int(row[3])
    
    if int(row[7]) == 1:
        prompt += "They have a very low standard of living. "
    elif int(row[7]) == 2:
            prompt += "They have a below-average standard of living. "
    elif int(row[7]) == 3:
        prompt += "They have an above-average standard of living. "
    elif int(row[7]) == 4:
        prompt += "They have a very high standard of living. "
    else: 
        raise NotImplementedError
        
    if int(row[8]) == 0:
        prompt += "They are exposed to the media very often. "
    elif int(row[8]) == 1:
        prompt += "They are not exposed to the media very often. "
    else: 
        raise NotImplementedError
    
    prompt += "What's the contraceptive method likely being used in this family?"
   
    completion = str(row["y"])
        
    return "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)


def data2text_feature_name_54(row, integer = False, label = True):
    prompt = "When we have " 
    for i in range(1,len(row)):
        prompt += "%s = %.2f, " % (FEATURE_NAMES_54[i-1].replace("_"," ").lower(), row[i-1])
    
    prompt += "What's type of this vehicle?"
   
    completion = str(int(row["y"]))
        
    return "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)

def df2jsonl_feat_name(df, filename, did,integer = False):
    fpath = os.path.join('data', filename)
    if did == 48:
        jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name_48, integer = integer), axis = 1).tolist())
        with open(fpath, 'w') as f:
            f.write(jsonl)
        return fpath
    elif did == 23:
        jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name_23, integer = integer), axis = 1).tolist())
        with open(fpath, 'w') as f:
            f.write(jsonl)
        return fpath
    elif did == 54:
        jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name_54, integer = integer), axis = 1).tolist())
        with open(fpath, 'w') as f:
            f.write(jsonl)
        return fpath
    else:
        raise NotImplementedError