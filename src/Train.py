#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
 
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Concatenate, Activation, Permute, Dot, TimeDistributed
from keras.models import Model, load_model
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from keras.callbacks import CSVLogger

from MyEncode import blo_encode_920

def ModelTrain(train_pep,train_network,train_affini,train_cate,test_pep,test_network,test_affini,test_cate,pattern,epoch = 200,pep_length = 9):
    
    # print("train_pep type:", type(train_pep))
    # print("train_pep shape:", train_pep.shape)
    # print("train_network type:", type(train_network))
    # print("train_network shape:", train_network.shape)
    # print("train_affini type:", type(train_affini))
    # print("train_affini shape:", train_affini.shape)
    # print("train_cate type:", type(train_cate))
    # print("train_cate shape:", train_cate.shape)
    # sys.exit(0)
    
    # 卷积和全连接层的大小
    filters, fc1_size, fc2_size, fc3_size= 256, 256, 64, 4

    # 卷积核个数 
    kernel_size = 2
    models = []

    # 接受蛋白质序列
    inputs_1 = Input(shape = (pep_length,21))

    # 接受网络度值向量
    inputs_3 = Input(shape = (8,1))

    # 对input1进行卷积，relu激活，dropout正则化，最大池化
    pep_conv = Conv1D(filters,kernel_size,padding = 'same',activation = 'relu',strides = 1)(inputs_1)
    pep_conv = Dropout(0.7)(pep_conv)
    pep_maxpool = MaxPooling1D(pool_size=1)(pep_conv)

    # 对input2进行类似操作
    flk_conv = Conv1D(filters,kernel_size,padding = 'same',activation = 'relu',strides = 1)(inputs_3)
    flk_conv = Dropout(0.7)(flk_conv)
    flk_maxpool = MaxPooling1D(pool_size=1)(flk_conv)

    # 将卷积层输出的特征展平为一维向量，有4个版本
    flat_pep_0 = Flatten()(pep_conv)
    flat_pep_1 = Flatten()(pep_conv)
    flat_pep_2 = Flatten()(pep_conv)
    flat_pep_3 = Flatten()(pep_conv)

    # 同理
    flat_flk_0 = Flatten()(flk_conv)
    flat_flk_1 = Flatten()(flk_conv)
    flat_flk_2 = Flatten()(flk_conv)
    flat_flk_3 = Flatten()(flk_conv)

    # 合并4个版本的 pep 和 flk ，得到 cat
    cat_0 = Concatenate()([flat_pep_0, flat_flk_0])
    cat_1 = Concatenate()([flat_pep_1, flat_flk_1])
    cat_2 = Concatenate()([flat_pep_2, flat_flk_2])
    cat_3 = Concatenate()([flat_pep_3, flat_flk_3])

    # cat 经过全连接层，并采用relu激活函数，得到4个输出
    fc1_0 = Dense(fc1_size,activation = "relu")(cat_0)
    fc1_1 = Dense(fc1_size,activation = "relu")(cat_1)
    fc1_2 = Dense(fc1_size,activation = "relu")(cat_2)
    fc1_3 = Dense(fc1_size,activation = "relu")(cat_3)

    # 将4个fc1合并为一个向量 merge_1
    merge_1 = Concatenate()([fc1_0, fc1_1, fc1_2,fc1_3])
    # merge_1 = Dropout(0.2)(merge_1)

    # 全连接层fc2，relu；全连接层f3，relu
    fc2 = Dense(fc2_size,activation = "relu")(merge_1)
    fc3 = Dense(fc3_size,activation = "relu")(fc2)


    # Attention 部分
    # 计算 pep_conv 和 flk_conv 在各个位置的注意力权重
    pep_attention_weights = Flatten()(TimeDistributed(Dense(1))(pep_conv))
    flk_attention_weights = Flatten()(TimeDistributed(Dense(1))(flk_conv))
    pep_attention_weights = Activation('softmax')(pep_attention_weights)
    flk_attention_weights = Activation('softmax')(flk_attention_weights)

    # permute 使得注意力权重的维度与相应的卷积特征维度一致
    pep_conv_permute = Permute((2,1))(pep_conv)
    flk_conv_permute = Permute((2,1))(flk_conv)

    # dot 将注意力权重应用到 pep_conv 和 flk_conv 上，得到经过注意力机制后的特征 pep_attention 和 flk_attention。
    pep_attention = Dot(-1)([pep_conv_permute, pep_attention_weights])
    flk_attention = Dot(-1)([flk_conv_permute, flk_attention_weights])

    # 把3个东西合并，得到最终的输出
    merge_2 = Concatenate()([pep_attention,flk_attention,fc3])
    # merge_2 = Dropout(0.2)(merge_2)

    # 训练
    # 结合度模型
    if pattern == 'bind':
        out = Dense(1,activation = "sigmoid")(merge_2)
        model = Model(inputs=[inputs_1, inputs_3],outputs=out)
        # adadelta = keras.optimizers.Adadelta(lr=1, rho=0.95, epsilon=None,decay=0.0)
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(learning_rate=0.005),
                      metrics=['mse'])

        log_file = "log/bind_train_log.csv"
        csv_logger = CSVLogger(log_file, append=True)

        train_log = model.fit([np.array(train_pep),np.array(train_network)],
                              np.array(train_affini),
                              batch_size=256,
                              epochs = epoch,
                              validation_data=([np.array(test_pep),np.array(test_network)],np.array(test_affini)),
                              verbose = 1,
                              callbacks=[csv_logger])
        model.save("model/bind.h5")
    
    # 免疫性模型
    if pattern == 'immunogenicity':
        out = Dense(2,activation = "sigmoid")(merge_2)
        model = Model(inputs=[inputs_1, inputs_3],outputs=out)
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['categorical_accuracy'])

        log_file = "log/immuno_train_log.csv"
        csv_logger = CSVLogger(log_file, append=True)

        train_log = model.fit([np.array(train_pep),np.array(train_network)],
                              np.array(train_cate),
                              batch_size=256,
                              epochs = epoch,
                              validation_data=([np.array(test_pep),np.array(test_network)],np.array(test_cate)),
                              verbose = 1,
                              callbacks=[csv_logger])
        model.save("model/immuno.h5")

def peptides(df):
    sequences = df['sequence'].values
    encoded_sequences = []
    for sequence in sequences:
        encoded_sequence = blo_encode_920(sequence)
        encoded_sequences.append(encoded_sequence)
    encoded_sequences_np = np.array(encoded_sequences)
    return encoded_sequences_np

def network(df,flag):
    if flag == "bind":
        selected_columns = df.iloc[:, 4:12]
    else:
        selected_columns = df.iloc[:, 5:13]
    rows_array = selected_columns.values
    final_array = []

    for row in rows_array:
        two_array = []
        for i in range(8):
            two_array.append(np.array([row[i]]))
        final_array.append(np.array(two_array))

    final_array = np.array(final_array)
    return final_array

def affinity(df):
    affi = df['affinity_'].values
    return affi

def cate(df):
    label_column = df["Label"]
    def transform_label(label):
        if label == 1:
            return [1, 0]
        elif label < 1:
            return [0, 1]
    transformed_labels = label_column.apply(transform_label)
    result_array = np.array(transformed_labels.tolist())
    return result_array

def save_npy():
    df_bind = pd.read_csv('../data/bind_all.csv')
    df_immuno = pd.read_csv('../data/immuno_all.csv')
    df_immuno['sequence_length'] = df_immuno['sequence'].apply(len)
    is_sequence_length_9 = df_immuno['sequence_length'] == 9
    df_immuno = df_immuno[is_sequence_length_9]
    df_immuno.drop(columns=['sequence_length'], inplace=True)

    df_immuno = df_immuno[~df_immuno['sequence'].str.contains('\+')]
    df_immuno = df_immuno[~df_immuno['sequence'].str.contains('\_')]

    df_bind_train = df_bind.sample(frac=0.7, random_state=42)
    df_bind_test = df_bind.drop(df_bind_train.index)
    df_immuno_train = df_immuno.sample(frac=0.7, random_state=42)
    df_immuno_test = df_immuno.drop(df_immuno_train.index)

    # bind
    np.save("NumpyArray/train_pep_bind.npy", peptides(df_bind_train) )
    np.save("NumpyArray/train_network_bind.npy", network(df_bind_train,"bind"))
    np.save("NumpyArray/train_affini.npy", affinity(df_bind_train))
    np.save("NumpyArray/test_pep_bind.npy", peptides(df_bind_test) )
    np.save("NumpyArray/test_network_bind.npy", network(df_bind_test,"bind"))
    np.save("NumpyArray/test_affini.npy", affinity(df_bind_test))

    # immuno
    np.save("NumpyArray/train_pep_immuno.npy", peptides(df_immuno_train) )
    np.save("NumpyArray/train_network_immuno.npy", network(df_immuno_train,"immuno"))
    np.save("NumpyArray/train_cate.npy", cate(df_immuno_train))
    np.save("NumpyArray/test_pep_immuno.npy", peptides(df_immuno_test))
    np.save("NumpyArray/test_network_immuno.npy", network(df_immuno_test,"immuno"))
    np.save("NumpyArray/test_cate.npy", cate(df_immuno_test))


def test_model(model_path, test_pep, test_network, test_affini, test_cate=None, pattern='bind'):
    
    model = load_model(model_path)

    if pattern == 'bind':
        predictions = model.predict([test_pep, test_network])
        mse = np.mean((predictions - test_affini)**2)
        print("Affinity test MSE:", mse)
    else:
        predictions = model.predict([test_pep, test_network])
        categorical_accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(test_cate, axis=1))
        print("Immunogenicity test accuracy:", categorical_accuracy)

def main():
    # save_npy()
    train_pep_bind = np.load("NumpyArray/train_pep_bind.npy", allow_pickle=True)    
    train_network_bind = np.load("NumpyArray/train_network_bind.npy", allow_pickle=True)
    train_affini = np.load("NumpyArray/train_affini.npy", allow_pickle=True)
    test_pep_bind = np.load("NumpyArray/test_pep_bind.npy", allow_pickle=True)
    test_network_bind = np.load("NumpyArray/test_network_bind.npy", allow_pickle=True)
    test_affini = np.load("NumpyArray/test_affini.npy", allow_pickle=True)

    train_pep_immuno = np.load("NumpyArray/train_pep_immuno.npy", allow_pickle=True)
    train_network_immuno = np.load("NumpyArray/train_network_immuno.npy", allow_pickle=True)
    train_cate = np.load("NumpyArray/train_cate.npy", allow_pickle=True)
    test_pep_immuno = np.load("NumpyArray/test_pep_immuno.npy", allow_pickle=True)
    test_network_immuno = np.load("NumpyArray/test_network_immuno.npy", allow_pickle=True)
    test_cate = np.load("NumpyArray/test_cate.npy", allow_pickle=True)

    # print(train_pep_bind.shape)
    # print(test_pep_bind.shape)
    # print(train_pep_immuno.shape)
    # print(test_pep_immuno.shape)

    ModelTrain(train_pep_bind,train_network_bind,train_affini,train_cate,test_pep_bind,test_network_bind,test_affini,test_cate,pattern ='bind',epoch = 200,pep_length = 9)
    ModelTrain(train_pep_immuno,train_network_immuno,train_affini,train_cate,test_pep_immuno,test_network_immuno,test_affini,test_cate,pattern ='immunogenicity',epoch = 200,pep_length = 9)
    test_model('model/bind.h5', test_pep_bind, test_network_bind, test_affini, test_cate, pattern='bind')
    test_model('model/immuno.h5', test_pep_immuno, test_network_immuno, test_affini, test_cate, pattern='immunogenicity')

main()
# save_npy()