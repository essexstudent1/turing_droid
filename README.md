# turing-droid

“This is only a foretaste of what is to come and only the shadow of what is going to be.”

                     - Alan Turing, the father of artificial intelligence.
                       Alan Turing was bullied into suicide by his own government for the crime of being gay.  
                
TuringDroid is an AI-powered chatbot designed to detect and intervene in anti-LGBTQ+ cyberbullying.

This artefact is a proof-of-concept software application developed in support of a MSc Cyber Securty thesis.

The chatbot provides the following functions:

1) Training

The chatbot uses a custom annotated dataset to train a deep learning model to do binary classification on text inputs, predicting whether the texts are anti-LGBT cyberbullying or not. 
The dataset used is described in more detail at https://www.kaggle.com/datasets/kw5454331/anti-lgbt-cyberbullying-texts. 

The model leverages GloVe word vectors. To run the application, the 100d pre-trained word vector file needs to be downloaded (https://nlp.stanford.edu/projects/glove/) 
and placed in the 'input' folder.

The model is based on a bidirectional long short term memory (BiLSTM) recurrent neural network.

2) Detection

The chatbot is designed to monitor a specific subreddit on Reddit via API for all new comments posted. 
Comments are sent to the deep learning model for evaluation. Comments predicted to be anti-LGBT cyberbullying are placed into a message queue for the response function.

3) Response

The response engine runs as multiple consumer threads. These threads read detected anti-LGBT cyberbullying messages from the message queue.
Cyberbullying messages are sent via API to a publicly available large language model (LLM) artificial intelligence system. The messages are sent
as part of a prompt requested that the LLM provide a reponse from the standpoint of a bystander seeking to provide support to the cyberbullying victim.
The response message from the LLM is then posted as a reply to the cyberbullying comment on Reddit.
