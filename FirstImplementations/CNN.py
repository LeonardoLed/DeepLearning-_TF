

data_path = ''

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

FILE_TFS = data_path + 'TFS.fasta'
FILE_NOTFS = data_path + 'NoTfs.fasta'

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

example_tokenized = tokenizer.texts_to_sequences([clean_seq_tfs[0]])
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
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization

model = Sequential()

model.add(Conv2D(128, (4, 21), input_shape = X_train_tensor.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Conv2D(128, (4, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Conv2D(128, (16, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(MaxPooling2D(pool_size = (979,1), strides = 1))

model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))

opt = keras.optimizers.Adam(learning_rate = 0.001)
model.compile(loss = 'binary_crossentropy',
              optimizer = opt,
              metrics = ['accuracy'])

model.summary()

#%% TRAINING

batch_size = 128
epochs = 10

h = model.fit(X_train_tensor, y_train_tensor,
                    batch_size = batch_size,
                    epochs = epochs,
                    verbose = 1,
                    validation_data = (X_valid_tensor, y_valid_tensor),
                    )

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
    plt.show()

show_history(h)


#%%
