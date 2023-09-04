#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Install required modules

get_ipython().run_line_magic('pip', 'install numpy')
get_ipython().run_line_magic('pip', 'install pandas')
get_ipython().run_line_magic('pip', 'install tensorflow')
get_ipython().run_line_magic('pip', 'install scikit-learn')
get_ipython().run_line_magic('pip', 'install nltk')
get_ipython().run_line_magic('pip', 'install keras')
get_ipython().run_line_magic('pip', 'install praw')
get_ipython().run_line_magic('pip', 'install psaw')
get_ipython().run_line_magic('pip', 'install python-dotenv')
get_ipython().run_line_magic('pip', 'install hugchat')
print('Module installation complete.')


# In[2]:


# Import critical libraries and modules

# Machne learning libraries and modules
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

# Reddit API libraries and modules
import praw
import psaw
import os
from psaw import PushshiftAPI
from dotenv import load_dotenv

# LLM libraries and modules
from hugchat import hugchat
from hugchat.login import Login

print('Library initalization complete.')


# In[15]:


### BEGIN TRAINING PROCESS
print('Initialization complete. Commencing training process.')

# Download common stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# In[16]:


# Read the training data
train= pd.read_csv('input/anti-lgbt-cyberbullying.csv')
train.head()


# In[17]:


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
#train['clean_text'] = train['clean_text'].apply(lambda x: make_lower(x))
train['clean_text'] = train['clean_text'].apply(lambda x: stemming(x))

train.head()


# In[18]:


# Tokenize the cleaned texts.
train['tokenized'] = train['clean_text'].apply(word_tokenize)
train.head()


# In[19]:


# Remove stopwords from text
train['no_stopwords'] = train['tokenized'].apply(
    lambda x: [word for word in x if word not in set(nltk.corpus.stopwords.words('english'))])

train['no_stopwords'] = [' '.join(map(str, l)) for l in train['no_stopwords']]

train.head()


# In[20]:


#Place pre-processed text into a list
train_texts = train['no_stopwords'].tolist()

# Open and process GloVe embeddings
embeddings_index = dict()
f = open('input/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


# In[21]:


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


# In[22]:


# Split the data into training and testing sets

x_train, x_test, y_train, y_test = train_test_split(
                    padded_text, train["anti_lgbt"], 
                    test_size=0.2, shuffle=True)



# In[23]:


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


# In[24]:


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


# In[25]:


# Fit the training data to the model
model.fit(x_train, y_train, callbacks=[early_stopping], epochs = 30, batch_size=32)


# In[26]:


# Develop predictions on the test data
y_pred = model.predict(x_test)


# In[27]:


# Report on the effectiveness of the model
pr, rc, thresholds = precision_recall_curve(y_test, y_pred)
crossover_index = np.max(np.where(pr == rc))
crossover_cutoff = thresholds[crossover_index]
#crossover_recall = rc[crossover_index]
#print("Crossover at {0:.2f} with recall {1:.2f}".format(crossover_cutoff, crossover_recall))
print('Training complete. Model effectiveness:')
print(classification_report(y_test, y_pred > crossover_cutoff))


# In[13]:


## BEGIN PREDICTION MODULE

#test_text = 'fuck off retarded faggot'
#test_text = 'i love all of you beautiful people'
#test_text = 'Hi there beautiful people!'

# Define the prediction function. 
# This function takes a single text input and returns either 0 (not cyberbullying) or 1 (cyberbullying)
def predict_cb(text):
    my_text = remove_URL(text)
    my_text = remove_emoji(my_text)
    my_text = remove_html(my_text)
    my_text = remove_punct(my_text)
    #test_text = make_lower(test_text)
    my_text = stemming(my_text)
    my_text = word_tokenize(my_text)
    filtered_text = [t for t in my_text if not t in stopwords.words("english")]
    filtered_text = ' '.join(filtered_text)
    tok.fit_on_texts([filtered_text])
    #print(filtered_text)
    test_seq =  tok.texts_to_sequences([filtered_text])
    #print(test_seq)
    my_padded_text = pad_sequences(test_seq, maxlen=max_len_text, padding='post')
    prediction = model.predict(my_padded_text)
    #print(predict)
    prediction = np.round(prediction).astype(int)
    return(prediction)

#print(int(predict_cb(test_text)))


# In[14]:


## BEGIN DETECTION MODULE

# Load environment variables containing credentials
load_dotenv('../.env')

#V erify that required credentials exist as environment variables
credential_vars = ["client_id", "client_secret", "sns_username", "sns_password", "llm_username", "llm_password"]
for var in credential_vars:
    if var not in os.environ:
        raise EnvironmentError("Required environment variable {} is not set.".format(var))

# Import API credentials
client_id = os.environ['client_id']
client_secret = os.environ['client_secret']
sns_username = os.environ['sns_username']
sns_password = os.environ['sns_password']
llm_username = os.environ['llm_username']
llm_password = os.environ['llm_password']

# Set bot name
user_agent = 'TuringDroid:1.0'

# Set name of the subreddit to monitor
target_community = 'TuringDroidTesting'


# In[15]:


# Define the SNS (social networking site) class

class SNS:
    def __init__(self):
        self.name = 'Reddit'

    # Define function to connect to the target social networking site (in this case Reddit)
    def connect(self):
        my_sns = praw.Reddit(
            client_id = client_id,
            client_secret = client_secret,
            username = sns_username,
            password = sns_password,
            user_agent = '<' + user_agent + ':by: /u/' + sns_username + '>'
        )
        self.connection = my_sns
        
    # Define function to connect to a specific group/community (in this case a subreddit)
    def set_community(self, request):
        my_community = self.connection.subreddit(request)
        self.community = my_community

    # Monitor for new comments
    # This method will loop forever or until interrupted
    def monitor(self):
        for comment in self.community.stream.comments(skip_existing=True):
            print('Found new comment:', comment.id)
            print('   Author:', comment.author)
            print('   Text:', comment.body)
            cb = int(predict_cb(comment.body))
            print('   Cyberbullying prediction:', cb)
        


# In[16]:


# Connect to target social networking site

# Initialize social networking site class
sns = SNS()

# Connect to the target sns
sns.connect()

# Set target commmunity within sns for monitoring
sns.set_community(target_community)

print('Connected to Reddit using username:', sns_username)
print('Monitoring subreddit:', target_community)


# In[17]:


sns.monitor()





# In[ ]:





# In[ ]:




