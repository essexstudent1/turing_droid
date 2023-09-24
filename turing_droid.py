#!/usr/bin/env python
# coding: utf-8

# In[ ]:


“”" TuringDroid “”"

## SETUP


# In[ ]:


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
get_ipython().run_line_magic('pip', 'install imblearn')
print('Module installation complete.')


# In[ ]:


# Import general modules
import re
import string
import tqdm
from threading import Thread
from queue import Queue

# Import Machne learning libraries and modules
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve

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

#from imblearn.over_sampling import SMOTE

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


# In[ ]:


# Define Classes


# In[ ]:


class Model:
    """ A class used to represent a machine learning model.

    Attributes:
        name: A descriptive name of the model. Not currently used.
        model: An instantiation of a Keras model class.

    Methods:
        build: Add layers and other hyperparamenters to the model.
        train: Train the model using training data.
        validate: Validate the model using test data.
        predict_cb: Ask the model for a cyberbullying prediction. 
    """
    
    def __init__(self):
        """ Initialize the class and create Keras model """
        self.name = 'BiLSTM'
        self.model = Sequential()

    def build(self):
        """ Add layers and other hyperparamenters to the model """
        self.model.add(Embedding(vocab_size, 100, 
                                 input_length=max_len_text, 
                                 embeddings_initializer=Constant(text_embedding_matrix), 
                                 trainable=False))
        self.model.add(Bidirectional(LSTM(64, dropout=0.4, recurrent_dropout=0.4)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.4))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.optimzer=Adam()
        self.model.compile(loss='binary_crossentropy', 
                           optimizer=self.model.optimzer, 
                           metrics=['accuracy', 'mae', f1_score])

    def train(self, input_data, target_data, callbacks, epochs, batch_size):
        """ Train the model by invoking the Keras model fit method.

        Args:
            input_data: A list of padded and vectorized texts.
            target_data: A list of classifications for the input data.
            callbacks: A list of callbacks for the Keras fit method.
            epochs: The number of training iterations to invoke.
            batch_size: The training batch size.

        Returns:
            The results of the Keras model fit method.
        """
        my_results = self.model.fit(input_data, target_data, callbacks=[callbacks], epochs = epochs, batch_size=batch_size)
        return my_results

    def validate(self, input_data):
        """ Generate predictions from the model for validation.

        Args:
            input_data: A list of padded and vectorized texts.
    
        Returns:
            The results of the Keras model predict method.
        """
        my_predictions = self.model.predict(input_data)
        return my_predictions

    def predict_cb(self, input_text):
        """ Make a cyberbullying prediction on a string.

        Args:
            input_text: A string for which a prediciton is requested.

        Returns:
            A boolean indicating the results of the prediction:
                0 = Not cyberbullying.
                1 = Cyberbullying.
        """"
        my_text = remove_url(input_text)
        my_text = remove_emoji(my_text)
        my_text = remove_html(my_text)
        my_text = remove_punct(my_text)
        my_text = stemming(my_text)
        my_text = word_tokenize(my_text)
        print('tokenized text', my_text)
        filtered_text = [t for t in my_text if not t in stopwords.words("english")]
        filtered_text = ' '.join(filtered_text)
        print('filtered text', filtered_text)
        tok.fit_on_texts([filtered_text])
        test_seq =  tok.texts_to_sequences([filtered_text])
        my_padded_text = pad_sequences(test_seq, maxlen=max_len_text, padding='post')
        print('padded:', my_padded_text)
        my_prediction = self.model.predict(my_padded_text)
        print('raw prediction: ', my_prediction)
        my_prediction = np.round(my_prediction).astype(int)
        print('rounded prediction: ', my_prediction)
        return my_prediction


# In[ ]:


class SNS:
    """ A class used to represent a social networking site (SNS) 

    Attributes:
        name: The name of the SNS. Not currently used.
        connection: An instantiation of the praw Reddit class.
        community: An instantiation of the praw subreddit method.

    Methods:
        connect: Create an instantion of the praw Reddit class.
        set_community: Invoke the praw subreddit method.
        monitor: Monitor the SNS for new comments.
        reply: Post a reply to a comment on the SNS.
    """
    
    def __init__(self):
        """ Initialize the class """
        self.name = 'Reddit'
        self.connection = None
        self.community = None

    def connect(self, client_id, client_secret, username, password, user_agent):
        """ Create an instantion of the praw Reddit class.

        Args:
            client_id: The client identifier for the API.
            client_secret: The client authenticator for the API.
            username: The username for the SNS.
            password: The password for the SNS.
            user_agent: The user agent to pass to the SNS.

        Returns:
            None
        """
        my_sns = praw.Reddit(
            client_id = client_id,
            client_secret = client_secret,
            username = username,
            password = password,
            user_agent = '<' + user_agent + ':by: /u/' + username + '>'
        )
        self.connection = my_sns

    def set_community(self, request):
        """ Connect to a specific group/community (in this case a subreddit)

        Args:
            request: The requested group/community/subreddit

        Returns:
            None
        """
        my_community = self.connection.subreddit(request)
        self.community = my_community

    def monitor(self, my_model, my_queue):
        """ Monitor for new comments. This method will monitor the
            class's set community for new comments. The comments will
            be sent to the indicated model's predict_cb method for a
            cyberbullying prediction. Comments predicted to be 
            cyberbullying will be placed in the requested message queue.

            Args:
                my_model: An instantiation of the Model class.
                my_queue: A message queue.

            Returns:
                None. This method will loop forever or until interrupted.
        """
        for comment in self.community.stream.comments(skip_existing=True):
            my_cb = int(my_model.predict_cb(comment.body))
            # Log comments and results
            print('New comment found. Text: ', comment.body)
            print('Cyberbullying prediction = ', my_cb)
            if my_cb == 1:
                my_queue.put(comment)

    def reply(self, my_comment, response):
        """ Post a response to a specific comment

            Arg:
                my_comment: An instantiation of a praw comment
                response: A string containing the requested response.
    
            Returns:
                The results of the praw comment reply method.
        """
        my_reply = my_comment.reply(response)
        return my_reply


# In[ ]:


# Define the DATASET class

class Dataset:

    # Initialize the class
    def __init__(self, file):
        self.train = pd.read_csv(file)

    # Clean the input column of the dataset and place cleaned text into the output column
    def clean(self, input_col, output_col):
        self.train[output_col] = self.train[input_col].apply(lambda x: remove_url(x))
        self.train[output_col] = self.train[output_col].apply(lambda x: remove_emoji(x))
        self.train[output_col] = self.train[output_col].apply(lambda x: remove_html(x))
        self.train[output_col] = self.train[output_col].apply(lambda x: remove_punct(x))
        self.train[output_col] = self.train[output_col].apply(lambda x: stemming(x))

    # Apply a word tokenize function to the specified column in the dataset, 
    # applied to the specified output column
    def tokenize(self, input_col, output_col):
        self.train[output_col] = self.train[input_col].apply(word_tokenize)

    # Remove stopwords from the specified column in the dataset, 
    # applied to the specified output column
    def remove_stopwords(self, input_col, output_col):
        self.train[output_col] = self.train[input_col].apply(
            lambda x: [word for word in x if word not in set(nltk.corpus.stopwords.words('english'))])
        self.train[output_col] = [' '.join(map(str, l)) for l in self.train[output_col]]


# In[ ]:


# Define the LLM (large language model) class

class LLM:

     # Initialize the class
    def __init__(self):
        self.name = 'HuggingChat'
        self.sign = None
        self.cookies = None
        self.chatbot = None

    # Connect to the target LLM
    def connect(self, username, password):
        self.sign = Login(username, password)
        self.cookies = self.sign.login()
        self.chatbot = hugchat.ChatBot(cookies=self.cookies.get_dict())

    # Query the LLM and return the response
    def query(self, text):
        conv_id = self.chatbot.new_conversation()
        self.chatbot.change_conversation(conv_id)
        my_response = self.chatbot.chat(text)
        return my_response


# In[ ]:


## DEFINE FUNCTIONS


# In[ ]:


# Function to remove URLs from text
def remove_url(text):
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
def stemming(text):
    my_snow = SnowballStemmer('english')
    a=[my_snow.stem(i) for i in word_tokenize(text) ]
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


# In[ ]:


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


# In[ ]:


## DEFINE THREADS


# In[ ]:


# Define responder thread
# This consumer thread will monitor the specified message queue for cyberbullying messages
# It will wait a random amount of time, then query the specified LLM for a response.
# The response will then be posted as a reply to the specified social networking site.
def responder(my_queue, my_llm, my_sns):
    while True:
        my_comment = my_queue.get()
        # Wait a random amount of time before proceeding to emulate a real bystander
        #time.sleep(random.randint(60, 360))
        my_query = "The following comment from Reddit has been detected as " + \
            "cyberbullying against an LGBT individual. " + \
            "Commment = {" + my_comment.body + "}. " + \
            "Provide a response to this comment as though you are a" + \
            "bystander who wants to show support for the victim, "+ \
            "with the primary goal of mitigating the impact of this " + \
            "cyberbullying comment on their mental health. " + \
            "Your response should be both empathetic and respectful. " + \
            "Your response should be no longer than ten sentences and written in " + \
            "a casual tone appropriate for the website Reddit. " + \
            "Your response should be written in the persona of " + \
            "a 30-year old person who lives in the USA, has a liberal" + \
            "arts education, is technically adept, " + \
            "and is a strong ally of the LGBT community."
        my_response = my_llm.query(my_query)
        print('Recieved response from LLM:', my_response)
        my_sns.reply(my_comment, my_response)
        my_queue.task_done()


# In[ ]:


## BEGIN MAIN APPLICATION


# In[ ]:


# Load files and set variables

# Download common stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Load environment variables containing credentials
load_dotenv('../.env')

#V erify that required credentials exist as environment variables
credential_vars = ["client_id", "client_secret", "sns_username", "sns_password", "llm_username", "llm_password"]
for var in credential_vars:
    if var not in os.environ:
        raise EnvironmentError("Required environment variable {} is not set.".format(var))

# Import API credentials
sns_client_id = os.environ['sns_client_id']
sns_client_secret = os.environ['sns_client_secret']
sns_username = os.environ['sns_username']
sns_password = os.environ['sns_password']
llm_username = os.environ['llm_username']
llm_password = os.environ['llm_password']

# Set bot name
sns_user_agent = 'TuringDroid:1.0'

# Set name of the subreddit to monitor
target_community = 'TuringDroidTesting'


# In[ ]:


### BEGIN TRAINING PROCESS


# In[ ]:


print('Initialization complete. Commencing training process.')


# In[ ]:


# Read the training data
training_data = Dataset('input/anti-lgbt-cyberbullying.csv')
training_data.train.head()


# In[ ]:


# Clean the dataset
training_data.clean('text', 'clean_text')
training_data.train.head()


# In[ ]:


# Tokenize the dataset
training_data.tokenize('clean_text', 'processed_text')
training_data.train.head()


# In[ ]:


# Remove stopwords from the dataset
training_data.remove_stopwords('processed_text', 'processed_text')
training_data.train.head()


# In[ ]:


#Place pre-processed text into a list
train_texts = training_data.train['processed_text'].tolist()

# Open and process GloVe embeddings
embeddings_index = dict()
f = open('input/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


# In[ ]:


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


# In[ ]:


# Split the data into training and testing sets

x_train, x_test, y_train, y_test = train_test_split(
                    padded_text, training_data.train["anti_lgbt"],
                    test_size=0.2, shuffle=True)


# In[ ]:


#smote = SMOTE()
#x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train.values)
#print(X_train_smote.shape, y_train_smote.shape)


# In[ ]:


# Define early stopping for the training model
early_stopping = EarlyStopping(
    monitor="loss",
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)


# In[ ]:


# Build and train the machine learning model

ml_model = Model()

ml_model.build()

ml_model.train(x_train, y_train, early_stopping, 100, 64)
#ml_model.train(x_train_smote, y_train_smote, early_stopping, 100, 64)


# In[ ]:


# Develop predictions on the test data
y_pred = ml_model.validate(x_test)


# In[ ]:


# Report on the effectiveness of the model
pr, rc, thresholds = precision_recall_curve(y_test, y_pred)
crossover_index = np.max(np.where(pr == rc))
crossover_cutoff = thresholds[crossover_index]
print('Training complete. Model effectiveness:')
print(classification_report(y_test, y_pred > crossover_cutoff))


# In[ ]:


## BEGIN DETECTION & RESPONSE MODULE


# In[ ]:


# Connect to target social networking site

# Initialize social networking site class
sns = SNS()

# Connect to the target sns
sns.connect(sns_client_id, sns_client_secret, sns_username, sns_password, sns_user_agent)

# Set target commmunity within sns for monitoring
sns.set_community(target_community)


# In[ ]:


# Connect to target large language model

# Initialize LLM class
llm = LLM()

# Connect to the target LLM
llm.connect(llm_username, llm_password)


# In[ ]:


# Set up message queue
MsgQ = Queue()


# In[ ]:


# Start four responder threads
thread_list = []
for i in range(4):
    thread = Thread(target=responder, args=(MsgQ, llm, sns,))
    thread_list.append(thread)
    thread.start()


# In[ ]:


# Begin monitoring
print('Detection module initialized.')
print('Connected to site ' + sns.name + ' using username ' + sns_username)
print('Connected to LLM ' + llm.name + ' using username ' + llm_username)
print('Starting to monitor community', target_community)
sns.monitor(ml_model, MsgQ)

# Never reached


# In[ ]:


#some testing stuff
#test_text = "Congratulations! I am so glad you are able to be yourself. Love is love and gender doesn't have to be binary."
test_text = 'this is retarded. there are only two genders, and pride is a sin. all queers go to hell.'
my_test = ml_model.predict_cb(test_text)
print(my_test)

