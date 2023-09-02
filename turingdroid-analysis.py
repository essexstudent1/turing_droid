#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Update core libraries 

get_ipython().system('pip install keras-core --upgrade')
get_ipython().system('pip install -q keras-nlp --upgrade')
#!pip install scipy --upgrade


# In[2]:


# Import critical libraries and modules

import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

import re
import string
import tqdm
import nltk

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import SnowballStemmer
from wordcloud import WordCloud

from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, LSTM, Dropout, BatchNormalization,SpatialDropout1D
from keras.layers import Embedding
from keras.initializers import Constant
from keras.optimizers import Adam
from keras import backend as K


# In[3]:


# Download and unzip (if required) NLTK stopwords

nltk.download('stopwords')
nltk.download('wordnet')
#!unzip /usr/share/nltk_data/corpora/wordnet.zip -d /usr/share/nltk_data/corpora/


# In[4]:


# Read dataset
train= pd.read_csv('../input/anti-lgbt-cyberbullying-texts/anti-lgbt-cyberbullying.csv')
train.head()


# In[5]:


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

def make_lower(text):
    return text.lower()

# Function to remove punctionation from text
def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

# Apply functions
train['nostem_text'] = train['text'].apply(lambda x: remove_URL(x))
train['nostem_text'] = train['nostem_text'].apply(lambda x: remove_emoji(x))
train['nostem_text'] = train['nostem_text'].apply(lambda x: remove_html(x))
train['nostem_text'] = train['nostem_text'].apply(lambda x: remove_punct(x))
train['nostem_text'] = train['nostem_text'].apply(lambda x: make_lower(x))
train['clean_text'] = train['nostem_text'].apply(lambda x: stemming(x))

train.head()


# In[6]:


# Data Analysis

# Visualize the target classes
plt.figure(figsize=(8,5))
plt.title("Count of Target Classes")
sns.countplot(y=train["anti_lgbt"],linewidth=2,
                   edgecolor='black')

plt.show()


# In[7]:


# Analyse number of words in text.

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
char_len_dis = train[train['anti_lgbt']==1]['text'].str.split().map(lambda x: len(x))
ax1.hist(char_len_dis,color='red',edgecolor='black', linewidth=1.2)
ax1.set_title('Anti-LGBT Cyberbullying Texts')
char_len_ndis = train[train['anti_lgbt']==0]['text'].str.split().map(lambda x: len(x))
ax2.hist(char_len_ndis,color='blue',edgecolor='black', linewidth=1.2)
ax2.set_title('Non-Cyberbullying Texts')
plt.suptitle("Length of words in text")
plt.tight_layout()
plt.show()


# In[8]:


# Tokenize the cleaned texts.
train['tokenized'] = train['clean_text'].apply(word_tokenize)
train['nostem_text'] = train['nostem_text'].apply(word_tokenize)
train.head()


# In[ ]:





# In[9]:


# Remove stopwords from text
train['no_stopwords'] = train['tokenized'].apply(
    lambda x: [word for word in x if word not in set(nltk.corpus.stopwords.words('english'))])

train['no_stopwords'] = [' '.join(map(str, l)) for l in train['no_stopwords']]

# Remove stopwords from un-stemmed text (used for wordcloud)
train['nostemnostop_text'] = train['nostem_text'].apply(
    lambda x: [word for word in x if word not in set(nltk.corpus.stopwords.words('english'))])

train['nostemnostop_text'] = [' '.join(map(str, l)) for l in train['nostemnostop_text']]

train.head()


# In[10]:


# Analyse top 50 words in training data

cb_text = train[train.anti_lgbt==1]["no_stopwords"]
non_cb_text = train[train.anti_lgbt==0]["no_stopwords"]

color = ['Paired','Accent']
splitedData = [cb_text,non_cb_text]
title = ["Anti-LGBT Cyberbullying Texts", "Non-Cyberbullying Texts"]
for item in range(2):
    plt.figure(figsize=(20,8))
    plt.title(title[item],fontsize=12)
    pd.Series(' '.join([i for i in splitedData[item]]).split()).value_counts().head(50).plot(kind='bar',fontsize=10,colormap=color[item],edgecolor='black', linewidth=1.2)
    plt.show()


# In[11]:


#Place pre-processed text into a list
train_texts = train['no_stopwords'].tolist()


# In[12]:


# Analysing common anti-LGBT words using WordCloud 

wc = WordCloud(background_color='white',width = 2048, height = 1080 )
wc.generate(' '.join(train[train.anti_lgbt==1]["nostemnostop_text"]))
plt.figure(figsize = (30,8))
plt.imshow(wc, interpolation="bilinear")

plt.axis('off')
plt.show()


# In[13]:


# Open and process GloVe embeddings
embeddings_index = dict()
f = open('../input/glove6b100dtxt/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


# In[14]:


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
        


# In[15]:


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
                    padded_text,train["anti_lgbt"], 
                    test_size=0.2, shuffle=True)


# In[16]:


# Test LogisticRegression
model_1 = LogisticRegression(C=1.0, max_iter=5000)
model_1.fit(x_train,y_train)
pred_1 = model_1.predict(x_test)
cr1    = classification_report(y_test,pred_1)
print(cr1)


# In[17]:


# Test Native Bayes
model_2 = MultinomialNB(alpha=0.7)
model_2.fit(x_train,y_train)
pred_2 = model_2.predict(x_test)
cr2    = classification_report(y_test,pred_2)
print(cr2)


# In[18]:


# Test Decision Trees
model_3 = DecisionTreeClassifier()
model_3.fit(x_train,y_train)
pred_3 = model_3.predict(x_test)
cr3    = classification_report(y_test,pred_3)
print(cr3)


# In[19]:


# Test Random Forest
model_4 = RandomForestClassifier()
model_4.fit(x_train,y_train)
pred_4 = model_4.predict(x_test)
cr4    = classification_report(y_test,pred_4)
print(cr4)


# In[20]:


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


# In[21]:


# Define early stopping callback for BiLSTM
early_stopping = EarlyStopping(
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


# In[22]:


# Fit the training data to the model
model.fit(x_train, y_train, callbacks=[early_stopping], epochs = 100, batch_size=32, verbose=1)


# In[23]:


# Develop predictions on the test data
y_pred = model.predict(x_test)


# In[24]:


# Report on the effectiveness of the model
pr, rc, thresholds = precision_recall_curve(y_test, y_pred)
plt.plot(thresholds, pr[1:])
plt.plot(thresholds, rc[1:])
plt.show()
crossover_index = np.max(np.where(pr == rc))
crossover_cutoff = thresholds[crossover_index]
crossover_recall = rc[crossover_index]
print("Crossover at {0:.2f} with recall {1:.2f}".format(crossover_cutoff, crossover_recall))
print(classification_report(y_test, y_pred > crossover_cutoff))

