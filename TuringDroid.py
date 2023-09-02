#!/usr/bin/env python
# coding: utf-8

# In[257]:


# Install required modules

get_ipython().run_line_magic('pip', 'install numpy')
get_ipython().run_line_magic('pip', 'install pandas')
get_ipython().run_line_magic('pip', 'install tensorflow')
get_ipython().run_line_magic('pip', 'install scikit-learn')
get_ipython().run_line_magic('pip', 'install nltk')
get_ipython().run_line_magic('pip', 'install keras')


# In[258]:


# Import critical libraries and modules

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve

import re
import string
import tqdm
import nltk

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import SnowballStemmer

from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, LSTM, Dropout, BatchNormalization,SpatialDropout1D
from keras.layers import Embedding
from keras.initializers import Constant
from keras.optimizers import Adam
from keras import backend as K


# In[259]:


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# In[260]:


train= pd.read_csv('~/input/anti-lgbt-cyberbullying.csv')
train.head()


# In[261]:


# Define pre-processing functions

# Function to remove URLs from text
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)

# Function to remove emojis from text
def remove_emoji(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Function to find word stems
snow = SnowballStemmer('english')
def stemming(string):
    a=[snow.stem(i) for i in word_tokenize(string) ]
    return " ".join(a)

# Function to remove HTML from text
def remove_html(text):
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', text)

# Function to make the text lowercase
def make_lower(text):
    return text.lower()

# Function to remove punctionation from text
def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

# Apply functions
train['clean_text'] = train['text'].apply(lambda x: remove_URL(x))
train['clean_text'] = train['clean_text'].apply(lambda x: remove_emoji(x))
train['clean_text'] = train['clean_text'].apply(lambda x: remove_html(x))
train['clean_text'] = train['clean_text'].apply(lambda x: remove_punct(x))
train['clean_text'] = train['clean_text'].apply(lambda x: make_lower(x))
train['clean_text'] = train['clean_text'].apply(lambda x: stemming(x))

train.head()


# In[262]:


# Tokenize the cleaned texts.
train['tokenized'] = train['clean_text'].apply(word_tokenize)
train.head()


# In[263]:


# Remove stopwords from text
train['no_stopwords'] = train['tokenized'].apply(
    lambda x: [word for word in x if word not in set(nltk.corpus.stopwords.words('english'))])

train['no_stopwords'] = [' '.join(map(str, l)) for l in train['no_stopwords']]

train.head()


# In[264]:


#Place pre-processed text into a list
train_texts = train['no_stopwords'].tolist()

# Open and process GloVe embeddings
embeddings_index = dict()
f = open('/home/arahu/input/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


# In[265]:


# Create word embeddings for the texts
max_len_text = 100

tok = Tokenizer()
tok.fit_on_texts(train_texts)
vocab_size = len(tok.word_index) + 1
encoded_text = tok.texts_to_sequences(train_texts)
padded_text = pad_sequences(encoded_text, maxlen=max_len_text, padding='post')

vocab_size = len(tok.word_index) + 1

text_embedding_matrix = np.zeros((vocab_size, 100))
for word, i in tok.word_index.items():
    t_embedding_vector = embeddings_index.get(word)
    if t_embedding_vector is not None:
        text_embedding_matrix[i] = t_embedding_vector


# In[277]:


# Split the data into training and testing sets
X = padded_text
Y = train["anti_lgbt"]

x_train, x_test, y_train, y_test = train_test_split(
                    X,Y, 
                    test_size=0.2, shuffle=True)



# In[278]:


# define the f1 score algorithm required for BiLSTM 
# from https://saturncloud.io/blog/adding-f1score-metric-in-compile-method-for-keras-a-comprehensive-guide/

def f1_score(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


# In[279]:


# Define early stopping callback for BiLSTM
early_stopping = EarlyStopping(
    monitor="loss",
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)

# Build the BiLSTM model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_len_text, embeddings_initializer=Constant(text_embedding_matrix), trainable=False))
model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
optimzer=Adam(learning_rate=3e-4)
model.compile(loss='binary_crossentropy', optimizer=optimzer, metrics=['accuracy', 'mae', f1_score])

print(model.summary())


# In[280]:


# Fit the training data to the model
model.fit(x_train, y_train, callbacks=[early_stopping], epochs = 30, batch_size=32, verbose=1)


# In[282]:


# Develop predictions on the test data
results = model.evaluate(x_test, y_test, batch_size=32)
print (results)
y_pred = model.predict(x_test)
#print(y_test[:5])
#print(y_pred[:5])


# In[283]:


# Report on the effectiveness of the model
pr, rc, thresholds = precision_recall_curve(y_test, y_pred)
#plt.plot(thresholds, pr[1:])
#plt.plot(thresholds, rc[1:])
#plt.show()
crossover_index = np.max(np.where(pr == rc))
crossover_cutoff = thresholds[crossover_index]
crossover_recall = rc[crossover_index]
print("Crossover at {0:.2f} with recall {1:.2f}".format(crossover_cutoff, crossover_recall))
print(classification_report(y_test, y_pred > crossover_cutoff))


# In[290]:


test_text = 'fuck off retarded faggot'
#test_text = 'i love all of you beautiful people'

test_text = remove_URL(test_text)
test_text = remove_emoji(test_text)
test_text = remove_html(test_text)
test_text = remove_punct(test_text)
#test_text = make_lower(test_text)
test_text = stemming(test_text)
test_text = word_tokenize(test_text)
filtered_text = [t for t in test_text if not t in stopwords.words("english")]
filtered_text = ' '.join(filtered_text)
#filtered_text = [' '.join(map(str, l)) for l in filtered_text]
#filtered_text = test_text
#print(" ".join(filtered_text))
#filtered_text = test_text
#filtered_text = join(filtered_text)
print(filtered_text)


# In[291]:


tok.fit_on_texts([filtered_text])
print(filtered_text)
test_seq =  tok.texts_to_sequences([filtered_text])
print(test_seq)

new_padded_text = pad_sequences(test_seq, maxlen=max_len_text, padding='post')

print(new_padded_text)



# In[292]:


predict = model.predict(new_padded_text)
print(predict)
predict = np.round(predict).astype(int)
print(predict)


# In[ ]:





# In[ ]:





# In[ ]:




