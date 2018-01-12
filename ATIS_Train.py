import data.load
import numpy as np
import keras
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = 3
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Conv1D
from keras.callbacks import ModelCheckpoint, TensorBoard
import progressbar

train_set, valid_set, dicts = data.load.atisfull()
w2idx, labels2idx = dicts['words2idx'], dicts['labels2idx']

train_x, _, train_label = train_set
val_x, _, val_label = valid_set

# Create index to word/label dicts
idx2w  = {w2idx[k]:k for k in w2idx}
idx2la = {labels2idx[k]:k for k in labels2idx}

# For conlleval script
words_train = [ list(map(lambda x: idx2w[x], w)) for w in train_x]
labels_train = [ list(map(lambda x: idx2la[x], y)) for y in train_label]
words_val = [ list(map(lambda x: idx2w[x], w)) for w in val_x]
labels_val = [ list(map(lambda x: idx2la[x], y)) for y in val_label]

n_classes = len(idx2la)
n_vocab = len(idx2w)

print("Example sentence : {}".format(words_train[1]))
print("Encoded form: {}".format(train_x[1]))
print()
print("It's label : {}".format(labels_train[1]))
print("Encoded form: {}".format(train_label[1]))

model = Sequential()
model.add(Embedding(n_vocab,100))
model.add(Conv1D(188,5,border_mode='same', activation='relu'))
model.add(Dropout(0.25))
model.add(LSTM(100,return_sequences=True))
model.add(TimeDistributed(Dense(n_classes, activation='softmax')))
model.compile('rmsprop', 'categorical_crossentropy')

#Callbacks <funny now hhhh>
import time
path = "output/"
checkpointer = ModelCheckpoint(path+"ATIS_LSTM-"+str(time.time())+".h5", verbose=0,
                          save_best_only=True)
tensorboard = TensorBoard(log_dir=path, write_images=True,
                      write_graph=True, histogram_freq=0)

n_epochs = 1
path = "output/"
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

    model.save_weights(path + "ATIS_LSTM-" + str(i) + ".h5")
    model_json = model.to_json()
    with open(path + "model_lstm.json", "w") as jf:
        jf.write(model_json)
