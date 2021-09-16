# Lecture3 NLP

## process texts
- Tokenizer

```Python
# Get the text and set the stop words
!gdown --id 1rX10xeI3eUJmOLsc4pOPY6AnCLO8DxNj

import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
#-------------------------------------------------------------------
# pre-process the source
sentences = []
labels = []
with open("./bbc-text.csv", 'r') as csvfile:
    ### START CODE HERE
    
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[0])
        sentence = row[1]
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
            sentence = sentence.replace("  ", " ")
        sentences.append(sentence)
        
    ### END CODE HERE


print(len(sentences))
print(sentences[0])
#--------------------------------------------------------------------
# set the tokenizer
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(len(word_index))

# sequences and do the padding
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
print(padded[0])
print(padded.shape)
#--------------------------------------------------------------------
# deal with the labels
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
label_word_index = label_tokenizer.word_index
label_seq = label_tokenizer.texts_to_sequences(labels)

print(label_seq)
print(label_word_index)
```

## words embedding
```Python
# Needed library
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#-----------------------------------------------------------------------
# Set the varibales
vocab_size = 1000 # used for num_words, the maximum number of words to keep
embedding_dim = 16
max_length = 120
trunc_type = 'post'
padding_type = 'post'
ovv_tok = "<ovv>"
training_portion = .8
#-----------------------------------------------------------------------
# set the lists
sentences = []
labels = []
# use stopwords to clean up unneeded words like "about"
stopwords = stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
#------------------------------------------------------------------------
# Deal with the source file
with open("PATH/file.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[0])
        sentence = row[1]
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
        sentences.append(sentence)
#---------------------------------------------------------------------------
# Divide the source into train and test
train_size = int(len(sentences) * training_portion)

train_sentences = sentences[:train_size]
train_labels = labels[:train_size]

validation_sentences = sentences[train_size:]
validation_labels = labels[train_size:]
#---------------------------------------------------------------------------
# initialize and fit the tokenizer for sentences
# sentences are just texts after canceling some words through stopwords
# sequences are list of numbers (a specific number is representing a word)
# paddeds are the sequences limited by max_len and padding_type(delete the pre numbers or the after numbers)
tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(train_sentences) # do the fit jobs
word_index = tokenizer.word_index # e.g. {'<OOV>':1, 's':2, ..., 'hello':23}
# build the train sequences
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)
# build the validation sequences
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
validation_padded = pad_sequences(validation_sequences, padding=padding_type,
maxlen=max_length)

# initialize and fit the tokenizer for labels
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))
#-----------------------------------------------------------------------------
# create, compile and build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

num_epochs = 30
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)
#------------------------------------------------------------------------------
# plot
import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
#------------------------------------------------------------------------------
# others stuffs
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# to see the shape
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)
#-------------------------------------------------------------------------------
# write and save the file
import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()
#--------------------------------------------------------------------------------
```

## RNN (Recurrent Neural Network)
- Sequence modeling
    - LSTM Long Short Term Memory
        - The pipeline context: Cell State (can be bidirectional e.g. later contexts can impact earlier one ) 
        ```Python
        # validation accuracy of 0.8142 after 10 epochs
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),

            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        ```

    - You can also use convolutional network

        ```Python
        # validation accuracy of 0.8192 after 10 epochs
        # fastest
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            
            tf.keras.layers.Conv1D(128, 5, activation='relu'),
            tf.keras.layers.GlobalMaxPooling1D(),

            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        ```

    - The Graded Recurrent Unit(GRU) is also a choice
        
        ```Python
        # validation accuracy of 0.8056 after 10 epochs
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim,
            input_length=max_length),

            tf.keras.layers.Biderectional(tf.keras.GRU(32)),

            tf.keras.layers.Dense(24, activation='relu),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        ```

- deal with large data set
```Python
model = tf.keras.Sequential([
    # the additional one for outer words
    # weights and trainable here because we used transfer learning
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length,
    weights=[embeddings_matrix], trainable=False)
    tf.keras.Dropout(0.2), # we usually do this in the transfer learning
    # the cnn part
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),

    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary
num_epochs = 10
history = model.fit(training_padded, training_labels, epochs=num_epochs, 
validation_data=(testing_padded, testing_labels), verbose=2)
```

## With literature

### to be noticed
```Python
# to keep all but not the last element
slice1 = list[:,:-1]
# to only keep the last element
slice2 = list[:,-1] 
```

```Python
# The points to increase the performance of network
model = Sequential()
# can change the embedding_dim (now is 100)
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
# can change the LSTM number
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))
# can change the optimizer (learning rate and algorithm)
adam = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(xs, ys, epochs=100, verbose=1)

```