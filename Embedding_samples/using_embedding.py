from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding

# define documents
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']

doc_v = ['well effort','paltry effort']

# define class labels
labels = [1,1,1,1,1,0,0,0,0,0]
labels_v = [1,0]

# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
print(vocab_size)

tv = Tokenizer()
tv.fit_on_texts(doc_v)
print(len(tv.word_index) + 1)

# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)

encoded_docsv = tv.texts_to_sequences(doc_v)
print(encoded_docsv)

# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
padded_docsv = pad_sequences(encoded_docsv, maxlen=max_length, padding='post')
print(padded_docs)
print(padded_docsv)

# load the whole embedding into memory
embeddings_index = dict()
f = open('/home/shenzhen/Downloads/glove.6B/glove.6B.50d.txt')

for line in f :
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()   

print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 50))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# time for defining the model
model = Sequential()
e = Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=4, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())

# Now, time for training the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# Now, lemme evaluate the model
loss, accuracy = model.evaluate(padded_docsv, labels_v, verbose=0)
print('Accuracy: %f :: loss: %f' %(accuracy%100, loss) )

model.predict(padded_docsv)

