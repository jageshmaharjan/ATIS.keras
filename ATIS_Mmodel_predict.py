import keras
import os
import numpy as np
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = 3
from keras.models import model_from_json
import data.load


train_set, valid_set, dicts = data.load.atisfull()
words2ids = dicts['words2idx']
labels2ids = dicts['labels2idx']
i = len(words2ids)
output = []

def word_exist(str):
    try:
        words2ids[str]
        return True
    except:
        return False

sentence = "i prefer to fly from atlanta to seattle"
sentenceList = sentence.split()
sentence2id = []
for str in sentenceList:
    if word_exist(str):
        sentence2id.append(words2ids[str])
    else:
        sentence2id.append(words2ids['<UNK>'])
        i = i+1;

print(sentence2id)


json_file = open('model_lstm.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('output_100epoch/ATIS_LSTM-99.h5')
print("model loaded from disk")
loaded_model.compile('rmsprop', 'categorical_crossentropy')
print("model")

pred = loaded_model.predict(np.array(sentence2id))
for i in range(len(pred)):
    output.append([key for key, value in labels2ids.items() if value == int(pred[i].argmax(axis=-1))])

print(output)