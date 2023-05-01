#!/usr/bin/env python
# coding: utf-8

# In[1]:


# execute this cell before you start

import tensorflow as tf
import tensorflow.keras as keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)
print(tf.keras.__version__)


# In[ ]:





# # Q2 The Reuters newswire data
# 
# Consider the data in  `tensorflow.keras.datasets.reuters` and train a network which reliably categorizes the newswires.  
# 
# Hints: 
# - some general explanations of all the datasets included in Keras is here: https://keras.io/datasets/
# After `from tensorflow.keras.datasets import reuters` you can get the dataset and the word index with through `reuters.get_word_index()` and `reuters.load_data()`.  The training labels correspond to different topics for each newswire.  The list of topics can be found here: https://github.com/keras-team/keras/issues/12072. 
# \
# 
# 
# 
# 

# ## Q2 The Reuters newswire data
# 

# ### <font color = blue> *Firtst, I download reuters dataset and created four numpy arrays which contains the data. The data is split into training and testing data.*

# In[2]:


reuters = keras.datasets.reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data()


# ### <font color = blue> *Check the data type, lengths, shape and look at each data set.*

# In[3]:


type(train_data), train_data.shape


# In[4]:


type(reuters.load_data())


# In[5]:


len(train_data[20]), len(train_data[21]), len(train_data[22]), len(train_data[23])


# In[6]:


train_data[20]


# ### <font color = blue> *Obtain the word index in order to understand which number corresponds to which word. At the same time, check the index data type, lengths and items.*

# In[7]:


word_index = reuters.get_word_index()


# In[8]:


type(word_index), len(word_index)


# In[9]:


word_index.items()


# ### <font color = blue> *Shifted by 3 and the numbers from 0 to 3 have special meanings.* 

# In[10]:


for k in word_index:
    word_index[k] += 3
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3


# In[11]:


word_index["good"], word_index["bad"]


# ### <font color = blue> *We need to build a reversed dictionary, which takes the numbers as keys and returns the words as values.* 

# In[12]:


reverse_word_index = {}
for k in word_index:
    reverse_word_index[word_index[k]] = k


# In[13]:


reverse_word_index = {word_index[k]:k for k in word_index}


# In[14]:


reverse_word_index[52], reverse_word_index[617],


# ### <font color = blue> *Convert all numbers in a given review back into the original text.*

# In[15]:


for i in train_data[20]:
    print(i,reverse_word_index.get(i,"?"))


# ### <font color = blue> *We use get(i,"?") instead of [i], to safeguard against the case, where a number is for some reason not in the dictionary.* 

# In[16]:


def decode_review(encoded_review):
    review = ""
    for k in encoded_review:
        review += " "
        review += reverse_word_index.get(k, '?')
    return review


# ### <font color = blue> *Check out a number of reviews and the corresponding recommendations:*

# In[17]:


decode_review(train_data[20]), train_labels[20]


# In[18]:


reverse_word_index[8000]


# ### <font color = blue> *Since we have a basic understanding of the content and type of data, so we begin to preprocess the data so that it could have a better performance in the network, we need to reduce the number of words in the data.*

# In[19]:


reuters = keras.datasets.reuters
vocab_size = 5000
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = vocab_size)


# In[20]:


vocab_size


# In[21]:


type(train_data),train_data.shape


# ### <font color = blue> *It is because the length of the individual reviews is not the same for all review, we need to normalize the input by cutting long reviews and expand short reviews by padding it.*

# In[22]:


train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=256)


# In[23]:


type(train_data), train_data.shape


# In[24]:


decode_review(train_data[20]), len(train_data[20])


# In[25]:


print(train_data[20])


# ### <font color = blue> *After preprocess data, we could build a model in tensorflow. It is necessary to select the appropriate node, activation function, and hidden layer To build a model to analyze the data. Without good accuracy, we cannot draw some convincing conclusions.* 

# In[26]:


vocab_size = 5000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 64))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(46, activation=tf.nn.softmax))
model.summary()


# ### <font color = blue> *My optimizer is adam and loss function is sparse_categorical_crossentropy,the target is required to be non-onehot encoding, and onehot encoding is implemented inside this function.*

# In[27]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])


# In[28]:


train_N=1000
x_val = train_data[:train_N]
partial_x_train = train_data[train_N:]

y_val = train_labels[:train_N]
partial_y_train = train_labels[train_N:]


# ### <font color = blue> *In this case, the data will be passed 40 times in the network during the training process, the number of samples per gradient update equal to 40, use validation_split=.1 instead of validation_data, which mean taking apart from the training set at a ratio of 0.1 as the validation set, then running the model.*

# In[29]:


# history = model.fit(partial_x_train, partial_y_train, epochs=40, 
# batch_size=512, validation_data=(x_val, y_val), verbose=1)
history = model.fit(train_data, train_labels, epochs=40, 
                    batch_size=512, validation_split=.1, verbose=1)


# ### <font color = blue> *Check the accuracy of the model, here we can see the accuracy is 65.72%.*

# In[30]:


model.evaluate(test_data, test_labels)


# ### <font color = blue> *Use history to view the parameters and types of the model.*

# In[31]:


history.params


# In[32]:


history_dict = history.history
history_dict.keys()


# In[33]:


type(history_dict["acc"]), len(history_dict["acc"])


# ### <font color = blue> *Consider the training loss and the validation loss, after about 5 training epochs, both losses have significantly improved. After about 25 training epochs, validation loss higher than training loss, there was some overfitting during training, I used a dropout layer in the network for regularization, but there was still a little overfitting.*

# ![image.png](attachment:image.png)

# In[34]:


acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = history.epoch

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


# ### <font color = blue> *At the same time, in the plot of training and validation accuracy, we could see that after the 5 training epochs, the accuracy rate is gradually increasing.*

# ![image.png](attachment:image.png)

# In[35]:


plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


# ### <font color = blue> *Calculate the direct output of the trained network for each data set in the test data and check the type of test labels:*

# In[36]:


model.evaluate(test_data, test_labels)


# In[37]:


predict = model.predict(test_data)
type(predict), predict.shape


# In[38]:


type(test_labels), test_labels.shape


# ### <font color = blue> *Calculate the error vector, by taking the difference between prediction and expected outcome and we can know which data set is correspond to error.*

# In[39]:


errors=[]
for i in range (45):
    errors = abs(predict[i]-test_labels[i])


# In[40]:


errors.shape


# In[41]:


max(errors)


# In[42]:


np.argmax(errors)


# ### <font color = blue> *the corresponding report is given by:*

# In[43]:


decode_review(test_data[np.argmax(errors)]), test_labels[np.argmax(errors)]


# ### <font color = blue> *We could get the review, which was predicted correctly with highest confidence with np.argmin and get the reviews in order from good to bad.*

# In[45]:


sortedindices = np.argsort(errors)
from ipywidgets import interact

@interact(n=(0,len(errors)-1))
def myf(n):
    i = sortedindices[n]
    print(f"index: {i}; test_label {test_labels[i]}; Prediction: {predict[i]}; error: {errors[i]}")
    print(decode_review(test_data[i]))
    


# In[ ]:





# In[ ]:




