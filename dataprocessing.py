import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from gensim.models import Word2Vec



def load_data(path):
    data = pd.read_csv(path,delimiter='\t',header=None)
    data.columns = ['sent_id','e1','e2','text']
    for col in ['e1','e2','text']:
        data[col] = data[col].map(lambda x:str(x).lower())
    return data


def load_relation(path, bag=False):
    label = pd.read_csv(path,delimiter='\t',header=None)
    if bag:
        label.columns = ['bag_id','e1','e2','sent_ids','label']
    else:
        label.columns = ['sent_id','label']
    return label


def load_w2v():
    w2v_model = Word2Vec.load('./w2v/w2v_model.w2v')
    return w2v_model


def get_entity_idx(data):
    data['e1_idx'] = data[['e1','text_seg']].apply(lambda x:x['text_seg'].index(x['e1']),axis=1)
    data['e2_idx'] = data[['e2','text_seg']].apply(lambda x:x['text_seg'].index(x['e2']),axis=1)
    return data


def get_pos_distance(data):
    data['e1_distance'] = data[['e1_idx','text_seg']].apply(lambda x:[i-x['e1_idx'] for i in range(len(x['text_seg']))],axis=1)
    data['e2_distance'] = data[['e2_idx','text_seg']].apply(lambda x:[i-x['e2_idx'] for i in range(len(x['text_seg']))],axis=1)
    return data


def get_word_index(text, index_word):
    idx_res = []
    for x in text:
        tmp = []
        for v in x:
            try:
                tmp.append(index_word[v])
            except KeyError:
                tmp.append(1)
        idx_res.append(tmp)
    return idx_res


def change_entity_idx(x):
    for i,v in enumerate(x['text_seg']):
        if v == x['e1']:
            x['word_index'][i] = 1
        if v == x['e2']:
            x['word_index'][i] = 1
    return x['word_index']


def modify_pos_idx(x):
    tmp = []
    for v in x:
        if v < 0:
            tmp.append(1)
        elif v > 99:
            tmp.append(99)
        else:
            tmp.append(v)
    return tmp


def get_sent_padding(data):
    data['word_index'] = data['word_index'].map(lambda x:x[:50])
    data['word_index'] = data['word_index'].map(lambda x:x + [0]*(50-len(x)))
    data['e1_distance'] = data['e1_distance'].map(lambda x:x[:50])
    data['e1_distance'] = data['e1_distance'].map(lambda x:x + [0]*(50-len(x)))
    data['e2_distance'] = data['e2_distance'].map(lambda x:x[:50])
    data['e2_distance'] = data['e2_distance'].map(lambda x:x + [0]*(50-len(x)))
    return data


def get_group_data(data):
    idx_grp = data.groupby(['e1','e2']).apply(lambda x:x['word_index'].values).reset_index()
    idx_grp.columns = ['e1','e2','word_idx']
    pos1_grp = data.groupby(['e1','e2']).apply(lambda x:x['e1_distance'].values).reset_index()
    pos1_grp.columns = ['e1','e2','e1_distance']
    pos2_grp = data.groupby(['e1','e2']).apply(lambda x:x['e2_distance'].values).reset_index()
    pos2_grp.columns = ['e1','e2','e2_distance']
    idx_grp  = idx_grp.merge(pos1_grp,on=['e1','e2'],how='left')
    idx_grp  = idx_grp.merge(pos2_grp,on=['e1','e2'],how='left')
    return idx_grp


def get_group_padding(data):
    data['word_idx'] = data['word_idx'].map(lambda x:x.tolist()+ [[0]*50]*(50-len(x)))
    data['e1_distance'] = data['e1_distance'].map(lambda x:x.tolist()+ [[0]*50]*(50-len(x)))
    data['e2_distance'] = data['e2_distance'].map(lambda x:x.tolist()+ [[0]*50]*(50-len(x)))
    data['word_idx'] = data['word_idx'].map(lambda x:x[:50])
    data['e1_distance'] = data['e1_distance'].map(lambda x:x[:50])
    data['e2_distance'] = data['e2_distance'].map(lambda x:x[:50])
    data['word_idx'] = data['word_idx'].map(lambda x:np.array(x))
    data['e1_distance'] = data['e1_distance'].map(lambda x:np.array(x))
    data['e2_distance'] = data['e2_distance'].map(lambda x:np.array(x))
    return data