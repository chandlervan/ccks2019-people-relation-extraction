import os
from collections import Counter
from gensim.models import Word2Vec
from generate_w2v import get_w2v
from dataprocessing import load_data, load_relation, get_entity_idx, get_pos_distance, get_word_index,\
    change_entity_idx, modify_pos_idx, get_sent_padding, get_group_data, get_group_padding
import numpy as np
import gc


if __name__ == '__main__':
    if os.path.exists('./w2v/'):
        pass
    else:
        get_w2v()
    gc.collect()
    # 载入基础数据
    w2v_model = Word2Vec.load('./w2v/w2v_model.w2v')
    sent_train = load_data('./open_data/sent_data_train.txt')  # 训练集
    sent_dev = load_data('./open_data/sent_data_dev.txt')  # 验证集
    sent_test = load_data('./open_data/sent_data_test.txt')  # 测试集
    train_label = load_relation('./open_data/sent_relation_train.txt')  # 训练标签
    dev_label = load_relation('./open_data/sent_relation_dev.txt')   # 验证标签
    sent_train = sent_train.merge(train_label, on='sent_id', how='left')
    sent_dev = sent_dev.merge(dev_label, on='sent_id', how='left')
    '''进行基础数据处理'''
    # 经过观察其实在sent_track这里一个句子只有一个关系，所以这样处理其实也没问题
    sent_train['label'] = sent_train['label'].map(lambda x: int(str(x).split(' ')[0]))
    sent_dev['label'] = sent_dev['label'].map(lambda x: int(str(x).split(' ')[0]))
    # 根据空格分词
    sent_train['text_seg'] = sent_train['text'].map(lambda x: str(x).lower().split(' '))
    sent_dev['text_seg'] = sent_dev['text'].map(lambda x: str(x).lower().split(' '))
    sent_test['text_seg'] = sent_test['text'].map(lambda x: str(x).split(' '))
    # 建立训练集的词典
    all_text_list = []
    for v in sent_train['text_seg'].values:
        all_text_list += v
    text_dict = Counter(all_text_list)
    new_text_dict = {key: text_dict[key] for key in text_dict.keys() if text_dict[key] >= 5}
    # 获取实体在序列中的位置，目前只标记位于第一个的位置，多次出现的暂无处理方法
    sent_train = get_entity_idx(sent_train)
    sent_dev = get_entity_idx(sent_dev)
    sent_test = get_entity_idx(sent_test)
    # 获取句子中其他词与实体之间的距离，实际中可能用位置向量较多，但实际意义相同
    sent_train = get_pos_distance(sent_train)
    sent_dev = get_pos_distance(sent_dev)
    sent_test = get_pos_distance(sent_test)
    # 判断哪些词在词向量模型中，因为要确定UNK 和 PAD
    word_in_w2v = []
    for key in new_text_dict.keys():
        if key in w2v_model:
            word_in_w2v.append(key)
    # 建立索引到词的映射
    word_index = dict()
    word_index[0] = 'PAD'
    word_index[1] = 'UNK'
    for i, word in enumerate(word_in_w2v):
        word_index[i + 2] = word
    index_word = {word_index[key]: key for key in word_index.keys()}
    # 建立词向量矩阵
    word_matrix = np.zeros((len(word_index), 128))
    for key in word_index:
        if word_index[key] not in ['UNK', 'PAD']:
            word_matrix[key] = w2v_model[word_index[key]]
    # 将字转换成索引
    sent_train['word_index'] = get_word_index(sent_train['text_seg'], index_word)
    sent_dev['word_index'] = get_word_index(sent_dev['text_seg'], index_word)
    sent_test['word_index'] = get_word_index(sent_test['text_seg'], index_word)
    # 为了避免模型学到错误的信息，将所有的实体都替换成1
    sent_train['word_index'] = sent_train[['e1','e2','text_seg','word_index']].apply(change_entity_idx, axis=1)
    sent_dev['word_index'] = sent_dev[['e1','e2','text_seg','word_index']].apply(change_entity_idx, axis=1)
    sent_test['word_index'] = sent_test[['e1','e2','text_seg','word_index']].apply(change_entity_idx, axis=1)
    # 将位置向量中大于预定长度的转换到合适的长度
    sent_train['e1_distance'] = sent_train['e1_distance'].map(modify_pos_idx)
    sent_train['e2_distance'] = sent_train['e2_distance'].map(modify_pos_idx)
    sent_dev['e1_distance'] = sent_dev['e1_distance'].map(modify_pos_idx)
    sent_dev['e2_distance'] = sent_dev['e2_distance'].map(modify_pos_idx)
    sent_test['e1_distance'] = sent_test['e1_distance'].map(modify_pos_idx)
    sent_test['e2_distance'] = sent_test['e2_distance'].map(modify_pos_idx)
    # 对于长度不足50的句子用0补充到50
    sent_train = get_sent_padding(sent_train)
    sent_dev = get_sent_padding(sent_dev)
    sent_test = get_sent_padding(sent_test)
    # 对于相同实体对的句子，进行集包处理
    train_idx_grp = get_group_data(sent_train)
    dev_idx_grp = get_group_data(sent_dev)
    test_idx_grp = get_group_data(sent_test)
    # 对于包中长度没有达到要求的，补上句子
    train_idx_grp = get_group_padding(train_idx_grp)
    dev_idx_grp = get_group_padding(dev_idx_grp)
    test_idx_grp = get_group_padding(test_idx_grp)
    # 为集包之后的训练集补上标签
    train_label_grp = sent_train.groupby(['e1', 'e2']).apply(lambda x: list(set(x['label'].values))[0]).reset_index()
    train_label_grp.columns = ['e1', 'e2', 'label']
    train_idx_grp = train_idx_grp.merge(train_label_grp, on=['e1', 'e2'], how='left')
    # 为集包之后的验证集补上标签
    dev_label_grp = sent_dev.groupby(['e1', 'e2']).apply(lambda x: list(set(x['label'].values))[0]).reset_index()
    dev_label_grp.columns = ['e1', 'e2', 'label']
    dev_idx_grp = dev_idx_grp.merge(dev_label_grp, on=['e1', 'e2'], how='left')


