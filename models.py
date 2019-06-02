import warnings
warnings.filterwarnings('ignore')
from keras.layers import Input,Dense,Lambda,concatenate,Embedding,BatchNormalization,\
    GlobalAveragePooling1D,Conv1D
from keras.models import Model
import keras.backend as K

def get_model(word_matrix):
    text_input = Input(shape=(50,50))
    pos1_input = Input(shape=(50,50))
    pos2_input = Input(shape=(50,50))
    text_weight = Embedding(len(word_matrix),128,weights=[word_matrix],mask_zero=True,trainable=True)
    pos1_emb = Embedding(100,8,mask_zero=True,trainable=True)
    pos2_emb = Embedding(100,8,mask_zero=True,trainable=True)
    all_text_vec = text_weight(text_input)
    all_pos1_vec = pos1_emb(pos1_input)
    all_pos2_vec = pos2_emb(pos2_input)
    all_vec = concatenate([all_text_vec,all_pos1_vec,all_pos2_vec])
    cnn1 = Conv1D(128,kernel_size=5)
    all_gru_vec = Lambda(lambda x:[K.expand_dims(cnn1(x[:,idx,:,:]),1) for idx in range(50)])(all_vec)
    all_gru_vec = concatenate(all_gru_vec,1)
    gm = Lambda(lambda x:[K.expand_dims(GlobalAveragePooling1D()(x[:,idx,:,:]),1) for idx in range(50)])(all_gru_vec)
    gru_vec = concatenate(gm,1)
    avg_1 = GlobalAveragePooling1D()(gru_vec)
    dense1 = Dense(64)
    dense_vec1 = dense1(avg_1)
    bn = BatchNormalization()
    dense_vec1 = bn(dense_vec1)
    ouput=Dense(35,activation='softmax')(dense_vec1)
    model = Model(input=[text_input, pos1_input, pos2_input],output=ouput)
    model.compile('nadam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model
