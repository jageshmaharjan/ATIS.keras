import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import progressbar
from keras.layers import Conv1D, Dropout, TimeDistributed, Dense
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers.embeddings import Embedding


data_path = '/home/jugs/PycharmProjects/ATIS.keras/preprocess/train_words'
label_path = '/home/jugs/PycharmProjects/ATIS.keras/preprocess/train_label'

def get_data_list(path):
    data_list =[]
    with open(path,'r') as f:
        for line in f:
            data_list.append(line)
    return data_list

def get_tokens_index(path):
    t = Tokenizer()
    corpus = get_data_list(path)
    t.fit_on_texts(corpus)
    data2idx = t
    vocab_size = len(t.word_index) + 1
    encoded = t.texts_to_sequences(corpus)
    return data2idx, encoded, vocab_size

def process_lbl(path):
    data_list = []
    data_index = dict()
    data_id_list = []
    with open(path,'r') as f:
        for line in f:
            data_list.append(line.split())
            for i, str in enumerate(line.split()):
                data_index[str] = i
    for line in data_list:
        new_line = []
        for i, str in enumerate(line):
            new_line.append(data_index[str])
        data_id_list.append(new_line)
    return data_index, data_id_list

traindata2idx, train_encoded, vocab_size = get_tokens_index(data_path)
label2idx, lbl_encoded = process_lbl(label_path)

# Create index to word/train_label dicts
idx2w  = {traindata2idx.word_index[k]:k for k in traindata2idx.word_index}
idx2la = {label2idx[k]:k for k in label2idx}
n_classes = len(idx2la)

words_train = [ list(map(lambda x: idx2w[x], w)) for w in train_encoded]
labels_train = [ list(map(lambda x: idx2la[x], y)) for y in lbl_encoded]
# words_val = [ list(map(lambda x: idx2w[x], w)) for w in val_x]  # validation data
# labels_val = [ list(map(lambda x: idx2la[x], y)) for y in val_label]    # validation data

# Creating a word embeeding from a Glove to Vec
embeddings_index = dict()
glove2vec = "/home/jugs/PycharmProjects/ATIS.keras/input/glove.6B.50d.txt"
f = open(glove2vec)
for line in f :
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# create a weight matrix for train_words in training docs
embedding_matrix = np.zeros((vocab_size, 50))
for word, i in traindata2idx.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print(len(embedding_matrix))


model = Sequential()
e = Embedding(vocab_size, 50, weights=[embedding_matrix], trainable=False)
model.add(e)
model.add(Conv1D(188,5,border_mode='same', activation='relu'))
model.add(Dropout(0.25))
model.add(LSTM(100,return_sequences=True))
model.add(TimeDistributed(Dense(n_classes, activation='softmax')))
model.compile('adam', 'categorical_crossentropy')


n_epochs = 2
path = "musicdata_output/"
for i in range(n_epochs):
    print("Training epoch {}".format(i))
    bar = progressbar.ProgressBar(max_value=len(train_encoded[:850]))
    for n_batch, sent in bar(enumerate(train_encoded[:850])):
        label = lbl_encoded[n_batch]
        # Make labels one hot .. not sure if its a good idea
        label = np.eye(n_classes)[label][np.newaxis, :]
        # View each sentence as a batch ..
        sent = sent[np.newaxis, :]

        if sent.shape[1] > 1:  # ignore 1 word sentences
            model.train_on_batch(sent, label)

    from metrics.accuracy import conlleval

    labels_pred_val = []
    bar = progressbar.ProgressBar(max_value=len(train_encoded[851:]))
    for n_batch, sent in bar(enumerate(train_encoded[851:])):
        label = lbl_encoded[n_batch]
        label = np.eye(n_classes)[label][np.newaxis, :]
        sent = sent[np.newaxis, :]

        pred = model.predict_on_batch(sent)
        pred = np.argmax(pred, -1)[0]
        labels_pred_val.append(pred)

    labels_pred_val = [list(map(lambda x: idx2la[x], y)) \
                       for y in labels_pred_val]
    con_dict = conlleval(labels_pred_val, labels_train[851:],
                         words_train[851:], 'measure.txt')

    print('Precision = {}, Recall = {}, F1 = {}'.format(
        con_dict['r'], con_dict['p'], con_dict['f1']))

    # model.fit(x=train_x, y=train_label, steps_per_epoch=1,callbacks=[checkpointer, tensorboard])

    model.save_weights(path + "music_LSTM-" + str(i) + ".h5")
    model_json = model.to_json()
    with open(path + "model_embed_lstm.json", "w") as jf:
        jf.write(model_json)