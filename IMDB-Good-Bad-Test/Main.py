#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np


# In[2]:


imdb=keras.datasets.imdb
(train_data,train_labels), (test_data, test_labels)=imdb.load_data(num_words=10000)

print(f"Training entries {len(train_data)}. Labels: {len(train_labels)}")
print(len(train_data[0]), len(train_data[1]))


# In[3]:


word_index=imdb.get_word_index()

word_index= {k:(v+3) for k,v in word_index.items()}
word_index["<pad>"]=0
word_index["<Start>"]=1
word_index["<unk>"]=2
word_index["<unused>"]=3


# In[4]:


reverse_word_index=dict([(value,key) for (key, value) in word_index.items()])

def decode_reveiw(text):
  return" ".join([reverse_word_index.get(i,"?") for i in text])


# In[6]:


print(train_data[0])
print(decode_reveiw(train_data[0]))


# In[7]:


train_data=keras.preprocessing.sequence.pad_sequences(train_data,
                                                     value=word_index["<pad>"],
                                                     padding="post",
                                                     maxlen=256)


test_data=keras.preprocessing.sequence.pad_sequences(test_data,
                                                     value=word_index["<pad>"],
                                                     padding="post",
                                                     maxlen=256)
print(len(train_data[0]), len(train_data[1]))
print(train_data[0])


# In[8]:


###neural network
vocab_size=10000
#inputlayer
model=keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
#hiddenlayer
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
#output layer
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))


model.summary()


# In[9]:


#loss function
model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["acc"])


x_val=train_data[:10000]
partial_x_train=train_data[10000:]


y_val=train_labels[:10000]
partial_y_train=train_labels[10000:]


# In[10]:


history=model.fit(partial_x_train,
                  partial_y_train,
                  epochs=75,
                  batch_size=75,
                  validation_data=(x_val, y_val),
                  verbose=1)


# In[11]:


results=model.evaluate(test_data,test_labels)
print(results)

