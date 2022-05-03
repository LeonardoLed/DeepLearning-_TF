# -*- coding: utf-8 -*-
"""
Created on 2022


"""

#%% COLAB
# !pip install biopython

# from google.colab import drive
# drive.mount('/content/drive')
# data_path = '/content/drive/MyDrive/Images/Leo_data/'

import time

#%% LOCAL PATH

data_path = './../Data/'

#%% DATA READING

import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split

def read_fasta_data(fasta_file):
    result = []
    seq_ids = []
    fp = open(fasta_file, 'r')
    for seq_record in SeqIO.parse(fp, 'fasta'):
        seq = seq_record.seq
        seq_id = seq_record.id
        result.append(str(seq))
        seq_ids.append(seq_id)
    fp.close()
    return result, seq_ids

FILE_TFS = data_path + 'TFS_corte.fasta'
FILE_NOTFS = data_path + 'NoTfs_corte.fasta'

seq_tfs, seq_tfs_ids = read_fasta_data(FILE_TFS)
seq_notfs, seq_notfs_ids = read_fasta_data(FILE_NOTFS)

#%% PLOT LENGTH GENERATOR

import matplotlib.pyplot as plt

lengths_tfs = [len(t) for t in seq_tfs]
plt.hist(lengths_tfs , bins = len(set(lengths_tfs)))
plt.title('Histogram of lengths TF')
plt.xlabel('length')
plt.xlim(0, 5000)
plt.show()

lengths_notfs = [len(t) for t in seq_notfs]
plt.hist(lengths_notfs , bins = len(set(lengths_notfs)))
plt.title('Histogram of lengths Non-TF')
plt.xlabel('length')
plt.xlim(0, 5000)
plt.show()

#%% Conteo de aminoacidos

def amino_count(sequences):

  amino_input = set()
  amino2count = {}

  for seq in sequences:

    seq_1 = list(seq)

    for amino in seq_1:
      if amino not in amino_input:
        amino_input.add(amino)
        amino2count[amino] = 1
      else:
        amino2count[amino] += 1

  return amino2count

amino2count_tf = amino_count(seq_tfs)
amino2count_notf = amino_count(seq_notfs)

print('Aminoacids present in tf dataset: ', amino2count_tf)
print('Aminoacids present in notf dataset: ', amino2count_notf)

#%% Gráficas histograma de aminoacidos

def histograms(amino2count_tf, amino2count_notf):

    import pandas as pd

    amino_tf = pd.DataFrame(list(amino2count_tf.items()) , columns = ['Amino', 'Values'])
    amino_tf = amino_tf.sort_values(by = ['Amino'], ascending = True)
    ax = amino_tf.plot.bar(x = 'Amino', y = 'Values', rot = 0, title = 'Aminoacids in TF')

    amino_notf = pd.DataFrame(list(amino2count_notf.items()) , columns = ['Amino', 'Values'])
    amino_notf = amino_notf.sort_values(by = ['Amino'])
    ax = amino_notf.plot.bar(x = 'Amino', y = 'Values', rot = 0, title = 'Aminoacids in noTF')

    return amino_tf

histograms(amino2count_tf, amino2count_notf)

#%% Limpieza de secuencias. Filtro de: (B,O,U,Z)

def cleaning_sequence(sequence):

    clean_seq_tfs = []

    for seq in sequence:

        seq_1 = list(seq)

        if not any(amino in seq_1 for amino in ('B','O','U','Z')):
            clean_seq_tfs.append(seq)

    return clean_seq_tfs

clean_seq_tfs = cleaning_sequence(seq_tfs)
clean_seq_notfs = cleaning_sequence(seq_notfs)

clean_amino2count_tf = amino_count(clean_seq_tfs)
clean_amino2count_notf = amino_count(clean_seq_notfs)

print('Aminoacids present in tf dataset: ', clean_amino2count_tf)
print('Aminoacids present in notf dataset: ', clean_amino2count_notf)

clean_amino_tf = histograms(clean_amino2count_tf, clean_amino2count_notf)

#%% Tokenization and example

from tensorflow.keras.preprocessing.text import Tokenizer

amino_names = list(clean_amino_tf['Amino'])
print('Aminoacids present in dataset: ', amino_names)
print('Number of different aminoacids: ', len(amino_names))

tokenizer = Tokenizer(num_words = len(amino_names), char_level = True)
tokenizer.fit_on_texts(amino_names)

print('Aminoacids tokenization: ', tokenizer.word_index)
print('Number of aminoacids tokenized: ', len(tokenizer.word_index))

example_tokenized = tokenizer.texts_to_sequences([clean_seq_tfs[0]]) #str to list
print('Example of tfs sequence: ', clean_seq_tfs[0])
print('Example of tfs sequence tokenized: ', example_tokenized)

#%% Zero padding and encoding

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def get_sequences(tokenizer , database):
  sequences_token = tokenizer.texts_to_sequences(database)
  sequences_padded = pad_sequences(sequences_token, truncating = 'post',
                                   padding = 'post', maxlen = 1000)
  sequences_encoded = to_categorical(sequences_padded)
  return sequences_encoded

padded_seq_tfs = get_sequences(tokenizer , clean_seq_tfs)

print('Size of full tfs dataset sequence padded: ', padded_seq_tfs.shape)
print('Length of padded example: ', len(padded_seq_tfs[0]))
# print('Example of tfs sequence padded: ', padded_seq_tfs[0])

padded_seq_notfs = get_sequences(tokenizer , clean_seq_notfs[0:35000])

print('Size of full notfs dataset sequence padded: ', padded_seq_notfs.shape)
print('Length of padded example: ', len(padded_seq_notfs[0]))
# print('Example of tfs sequence padded: ', padded_seq_notfs[0])

#%% Concatenation and generating labels
# WARNING OF CRASH OF RAM !!!!!!!!!!

# padded_seq_tfs = padded_seq_tfs[0:2000]
# padded_seq_notfs = padded_seq_notfs[0:2000]
seq_full = np.concatenate((padded_seq_tfs , padded_seq_notfs))
print('Size of full database concatenated: ', seq_full.shape)

labels_ones = np.ones((padded_seq_tfs.shape[0],), dtype = int)
labels_zeros = np.zeros((padded_seq_notfs.shape[0],), dtype = int)
labels = np.concatenate ((labels_ones , labels_zeros))
print('Size of labels vector: ', labels.shape)

#%% DATA SPLIT

X_train, X_valid, y_train, y_valid = train_test_split(seq_full, labels, test_size = 0.2,
                                                      random_state = 1, shuffle = True)
print('X_train shape:',X_train.shape, type(X_train))
print('y_train shape:',y_train.shape, type(y_train))

print('X_valid shape:',X_valid.shape, type(X_valid))
print('y_valid shape:',y_valid.shape, type(y_valid))


#%% CLEANING RAM

del seq_full

#%% TENSOR COVERSION

import tensorflow as tf
X_train_tensor = tf.expand_dims(X_train, -1)
X_valid_tensor = tf.expand_dims(X_valid, -1)
y_train_tensor = tf.expand_dims(y_train, -1)
y_valid_tensor = tf.expand_dims(y_valid, -1)

print('X_train_tensor shape:',X_train_tensor.shape, type(X_train_tensor))
print('y_train_tensor shape:',y_train_tensor.shape, type(y_train_tensor))

print('X_valid_tensor shape:',X_valid_tensor.shape, type(X_valid_tensor))
print('y_valid_tensor shape:',y_valid_tensor.shape, type(y_valid_tensor))

#%% CLEANING RAM

del X_train, X_valid, y_train, y_valid

#%% CNN MODEL BASED ON DEEPTFACTOR

from tensorflow import keras
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Dropout, Activation, Flatten, Concatenate, concatenate
from keras.layers import Conv2D, MaxPooling2D, MaxPooling3D, GlobalMaxPooling3D, GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras.utils.vis_utils import plot_model

###################### FIRST CNN #####################################
input_1 = Input(shape = X_train_tensor.shape[1:])

conv_11 = Conv2D(128, (4, 21))(input_1)
batch_11 = BatchNormalization()(conv_11)
activation_11 = Activation('relu')(batch_11)
drop_11 = Dropout(0.1)(activation_11)

conv_12 = Conv2D(128, (4, 1))(drop_11)
batch_12 = BatchNormalization()(conv_12)
activation_12 = Activation('relu')(batch_12)
drop_12 = Dropout(0.1)(activation_12)

conv_13 = Conv2D(128, (16, 1))(drop_12)
batch_13 = BatchNormalization()(conv_13)
activation_13 = Activation('relu')(batch_13)
drop_13 = Dropout(0.1)(activation_13)

###################### SECOND CNN #####################################
input_2 = Input(shape = X_train_tensor.shape[1:])

conv_21 = Conv2D(128, (12, 21))(input_2)
batch_21 = BatchNormalization()(conv_21)
activation_21 = Activation('relu')(batch_21)
drop_21 = Dropout(0.1)(activation_21)

conv_22 = Conv2D(128, (8, 1))(drop_21)
batch_22 = BatchNormalization()(conv_22)
activation_22 = Activation('relu')(batch_22)
drop_22 = Dropout(0.1)(activation_22)

conv_23 = Conv2D(128, (4, 1))(drop_22)
batch_23 = BatchNormalization()(conv_23)
activation_23 = Activation('relu')(batch_23)
drop_23 = Dropout(0.1)(activation_23)

###################### THIRD CNN #####################################
input_3 = Input(shape = X_train_tensor.shape[1:])

conv_31 = Conv2D(128, (12, 21))(input_3)
batch_31 = BatchNormalization()(conv_31)
activation_31 = Activation('relu')(batch_31)
drop_31 = Dropout(0.1)(activation_31)

conv_32 = Conv2D(128, (8, 1))(drop_31)
batch_32 = BatchNormalization()(conv_32)
activation_32 = Activation('relu')(batch_32)
drop_32 = Dropout(0.1)(activation_32)

conv_33 = Conv2D(128, (4, 1))(drop_32)
batch_33 = BatchNormalization()(conv_33)
activation_33 = Activation('relu')(batch_33)
drop_33 = Dropout(0.1)(activation_33)

###################### CONCATENATION #####################################

merge = concatenate([drop_13, drop_23, drop_33])

conv_4 = Conv2D(348, (1, 1))(merge)
batch_4 = BatchNormalization()(conv_4)
activation_4 = Activation('relu')(batch_4)

max_final = GlobalMaxPooling2D()(activation_4)
flat_f1 = Flatten()(max_final)
dense_f1 = Dense(512)(flat_f1)
batch_f1 = BatchNormalization()(dense_f1)
activation_f1 = Activation('relu')(batch_f1)

dense_f2 = Dense(1)(activation_f1)
batch_f2 = BatchNormalization()(dense_f2)
output = Activation('sigmoid')(batch_f2)

###################### MODEL #####################################

model = Model(inputs = [input_1, input_2, input_3], outputs = output)
opt = keras.optimizers.Adam(learning_rate = 0.001)
model.compile(loss = 'binary_crossentropy',
              optimizer = opt,
              metrics = ['accuracy'])

model.summary()

###################### SAVING SUMMARY ###################################
from contextlib import redirect_stdout

with open('model_summary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

plot_model(model, to_file = 'model_figure.png')


#%% TRAINING

batch_size = 32
epochs = 10
t0 = time.time()

h = model.fit([X_train_tensor,X_train_tensor,X_train_tensor], y_train_tensor,
                    batch_size = batch_size,
                    epochs = epochs,
                    verbose = 1,
                    validation_data = ([X_valid_tensor,X_valid_tensor,X_valid_tensor], y_valid_tensor),
                    )
t1 = time.time()

model.save('DeepTFactor_1.h5')

#%% DISPLAYING PLOTS

def show_history(h):
    epochs_trained = len(h.history['loss'])
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(0, epochs_trained), h.history.get('accuracy'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_accuracy'), label='Validation')
    plt.ylim([0., 1.])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(0, epochs_trained), h.history.get('loss'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_loss'), label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('CNN_history50.png')

    plt.show()

show_history(h)


#%%
