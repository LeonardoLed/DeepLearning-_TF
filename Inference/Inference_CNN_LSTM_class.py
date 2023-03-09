# -*- coding: utf-8 -*-

import numpy as np
from Bio import SeqIO
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model


#%%

class Data_inference():

    def __init__(self, data_path):

        self.data_path = data_path

    ##%%
    def loading(self, file_path):

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

        seq_test , seq_test_ids = read_fasta_data(file_path)

        return seq_test, seq_test_ids

    ##%%
    def cleaning(self, seq_test):

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

        def cleaning_sequence(sequence):

            clean_seq_tfs = []

            for seq in sequence:
                seq_1 = list(seq)
                if not any(amino in seq_1 for amino in ('B','O','U','Z')):
                    clean_seq_tfs.append(seq)

            return clean_seq_tfs

        clean_seq_test = cleaning_sequence(seq_test)

        clean_amino2count_test = amino_count(clean_seq_test)
        print('Aminoacids present in testing dataset after cleaning: ', clean_amino2count_test)
        print('Total of aminoacids in dataset: ', len(clean_amino2count_test))

        return clean_amino2count_test, clean_seq_test

    ##%%
    def counting(self, clean_amino2count_test):

        def histograms(amino2count_tf):

            import pandas as pd

            amino_tf = pd.DataFrame(list(amino2count_tf.items()) , columns = ['Amino', 'Values'])
            amino_tf = amino_tf.sort_values(by = ['Amino'], ascending = True)
            ax = amino_tf.plot.bar(x = 'Amino', y = 'Values', rot = 0, title = 'Aminoacids in test sequence')

            return amino_tf

        clean_amino_test = histograms(clean_amino2count_test)
        amino_names = list(clean_amino_test['Amino'])
        print('Aminoacids present in dataset: ', amino_names)
        print('Number of different aminoacids: ', len(amino_names))

        return amino_names


    ##%%
    def tokens(self, clean_seq_test):

        with open( 'tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            print(tokenizer)

        def get_sequences(tokenizer, database):
            sequences_token = tokenizer.texts_to_sequences(database)
            sequences_padded = pad_sequences(sequences_token, truncating = 'post',
                                             padding = 'post', maxlen = 1000)
            sequences_encoded = to_categorical(sequences_padded, num_classes = 21)
            return sequences_encoded

        padded_seq_test = get_sequences(tokenizer, clean_seq_test)

        return padded_seq_test


    ##%%
    def inference(self, padded_seq_test):

        model_loaded = load_model('saved_model.h5')
        model_loaded.summary()

        print('Testing sequence shape:', padded_seq_test.shape, type(padded_seq_test))

        import tensorflow as tf
        padded_seq_test = tf.expand_dims(padded_seq_test, -1)
        print('Expanded testing sequence shape:', padded_seq_test.shape, type(padded_seq_test))

        pred = model_loaded.predict([padded_seq_test, padded_seq_test, padded_seq_test, padded_seq_test])
        predictions = np.array((pred > 0.5).astype(np.uint8))
        positions = np.where(pred > 0.5)
        print('Total sequences: ', len(predictions))
        print('Predicted sequences with TF: ', np.sum(predictions))
        print('Predicted sequences with noTF: ', np.sum(1 - predictions))

        return predictions, positions, pred


def get_tfs(valors, indices, map):
    outfile = open(data_path + "predictions.txt", 'a')

    for reg in indices[0]:
        name_protein = map[reg]
        outfile.write(name_protein + "\t" + str(valors[reg])+"\n")

    outfile.close()


##%% MAIN

if __name__ == "__main__":

    data_path = './Conocidos/'
    file_path = data_path + 'Entraf.fas'

    data_inference = Data_inference(data_path)

    seq_test, map = data_inference.loading(file_path)
    clean_amino2count_test, clean_seq_test = data_inference.cleaning(seq_test)
    #amino_names = data_inference.counting(clean_amino2count_test)
    padded_seq_test = data_inference.tokens(clean_seq_test)
    predictions, positions, pred = data_inference.inference(padded_seq_test)
    get_tfs(pred, positions, map)
