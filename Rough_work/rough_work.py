import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import progressbar
from keras.layers import Conv1D, Dropout, TimeDistributed, Dense
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers.embeddings import Embedding

# embeddings_index = dict()
# glove2vec = "/home/jugs/PycharmProjects/ATIS.keras/input/glove.6B.50d.txt"
# f = open(glove2vec)
# for line in f :
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()


train_path = '/home/jugs/PycharmProjects/ATIS.keras/preprocess/train_words'
train_label_path = '/home/jugs/PycharmProjects/ATIS.keras/preprocess/train_label'
val_path = '/home/jugs/PycharmProjects/ATIS.keras/preprocess/val_words'
val_label_path = '/home/jugs/PycharmProjects/ATIS.keras/preprocess/val_label'

def processing(fpath):
    f = open(fpath)
    all_data_dict = dict()
    all_data = []
    i = 0
    for line in f:
        all_data.append(line.split())
        for str in line.split():
            if str not in all_data_dict:
                all_data_dict[str] = i
                i = i+1

    all_idx_data = []
    for line in all_data:
        tmp_idx = []
        for str in line:
            tmp_idx.append(all_data_dict[str])
        all_idx_data.append(np.array(tmp_idx))
    return all_data, all_idx_data, all_data_dict


words_train, train_x, w2idx = processing(train_path)
labels_train, train_label, labels2idx = processing(train_label_path)
words_val, val_x, val_x_dict = processing(val_path)
labels_val, val_label, val_label_dict = processing(val_label_path)

n_vocab = len(w2idx)
n_classes = len(labels2idx)

# Create index to word/train_label dicts
idx2w  = {w2idx[k]:k for k in w2idx}
idx2la = {labels2idx[k]:k for k in labels2idx}

# create a weight matrix for train_words in training docs
# embedding_matrix = np.zeros((len(train_x_dict), 50))
# for word, i in train_x_dict.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector

print("pre-processed")

model = Sequential()
model.add(Embedding(n_vocab, 50))
model.add(Conv1D(188,5,border_mode='same', activation='relu'))
model.add(Dropout(0.25))
model.add(LSTM(100,return_sequences=True))
model.add(TimeDistributed(Dense(n_classes, activation='softmax')))
model.compile('rmsprop', 'categorical_crossentropy')

n_epochs = 5
path = "/home/jugs/PycharmProjects/ATIS.keras/musicdata_output/"
for i in range(n_epochs):
    print("Training epoch {}".format(i))

    bar = progressbar.ProgressBar(max_value=len(train_x))
    for n_batch, sent in bar(enumerate(train_x)):
        label = train_label[n_batch]
        # Make labels one hot .. not sure if its a good idea
        label = np.eye(n_classes)[label][np.newaxis, :]
        # View each sentence as a batch ..
        sent = sent[np.newaxis, :]
        if sent.shape[1] > 1:  # ignore 1 word sentences
            model.train_on_batch(sent, label)

    from metrics.accuracy import conlleval

    labels_pred_val = []
    bar = progressbar.ProgressBar(max_value=len(val_x))
    for n_batch, sent in bar(enumerate(val_x)):
        label = val_label[n_batch]
        label = np.eye(n_classes)[label][np.newaxis, :]
        sent = sent[np.newaxis, :]

        pred = model.predict_on_batch(sent)
        pred = np.argmax(pred, -1)[0]
        labels_pred_val.append(pred)

    labels_pred_val = [list(map(lambda x: idx2la[x], y)) \
                       for y in labels_pred_val]
    con_dict = conlleval(labels_pred_val, labels_val,
                         words_val, 'measure.txt')

    print('Precision = {}, Recall = {}, F1 = {}'.format(
        con_dict['r'], con_dict['p'], con_dict['f1']))

    # model.fit(x=train_x, y=train_label, steps_per_epoch=1,callbacks=[checkpointer, tensorboard])

    model.save_weights(path + "MUSIC_LSTM-" + str(i) + ".h5")
    model_json = model.to_json()
    with open(path + "model_embed_lstm.json", "w") as jf:
        jf.write(model_json)