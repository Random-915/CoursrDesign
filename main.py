import tensorflow
import pandas as pd
import numpy as np
import jieba
import jieba.posseg as pseg
import re
import csv
import string
from keras import models
from keras import layers
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPool1D
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import word2vec
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import Model
from sklearn import metrics
from keras.models import load_model


train_data = pd.read_csv('trainNews1.csv', lineterminator="\n")


def encodeLabel(data):
    listLabel = []
    for label in data['channelName']:

        listLabel.append(label)
    le = LabelEncoder()
    resultLabel = le.fit_transform(listLabel)
    return resultLabel


def getReview(data):
    listReview = []
    for label in data['content']:
        listReview.append(label)
    return listReview


# if __name__ == '__main__':
trainLabel = encodeLabel(train_data)
trainReview = getReview(train_data)
# print(trainLabel)
# print(trainReview)

def stopwordslist():
    stopwordlist=[line.strip() for line in open('中文停用词表.txt',encoding='UTF-8').readlines()]
    return stopwordlist


def deleteStop(sentence):
    stopwords=stopwordslist()
    outstr=""
    for i in sentence:
        if i not in stopwords and i!="\n":
            outstr+=i
    return outstr

def wordCut(Review):
    Mat=[]
    for rec in Review:
        senten=[]

        rec=re.sub('[%s]' % re.escape(string.punctuation), '', str(rec))
        fenci=jieba.lcut(rec)
        stc=deleteStop(fenci)
        seg_list=pseg.cut(stc)
        for word,flag in seg_list:
            if flag not in ["f"]:
                senten.append(word)
        Mat.append(senten)
    return Mat
trainCut=wordCut(trainReview)





fileDic=open('wordCut.txt','w',encoding='UTF-8')
for i in trainCut:
    fileDic.write(" ".join(i))
    fileDic.write('\n')
fileDic.close()
words=[line.strip().split(" ") for line in open('wordCut.txt',encoding='UTF-8').readlines()]

maxLen=300




num_featrues=300
min_word_count=3
num_workers=4
context=4
model=word2vec.Word2Vec(trainCut,workers=num_workers,vector_size=num_featrues,min_count=min_word_count,window=context)
model.init_sims(replace=True)
model.save("CNNw2vModelFenci2")
model.wv.save_word2vec_format("CNNVectorFenci2",binary=False)
# print(model)
w2v_model=word2vec.Word2Vec.load("CNNw2vModelFenci2")
tokenizer=Tokenizer()
tokenizer.fit_on_texts(words)
vocab=tokenizer.word_index
trainID=tokenizer.texts_to_sequences(trainCut)
trainSeq=pad_sequences(trainID,maxlen=maxLen)
trainCate=to_categorical(trainLabel,num_classes=9)

# print(trainSeq)
# print(trainCate)
#
#
#
#
#
from keras.models import Sequential

embedding_matrix = np.zeros((len(vocab)+1, 300))
for word, i in vocab.items():
    try:
        embedding_vector=w2v_model.wv[str(word)]
        embedding_matrix[i]=embedding_vector
    except KeyError:
        continue
print(len(embedding_matrix))



# # import Input as Input
#
main_input = Input(shape=(maxLen,),dtype='float64')
embedder = Embedding(len(vocab)+1,300,input_length=maxLen,weights=[embedding_matrix],trainable=False)
model=Sequential()
model.add(embedder)
model.add(Conv1D(256,3,padding='same',activation='relu'))
model.add(MaxPool1D(50,3,padding='same'))
model.add(Conv1D(32,3,padding='same',activation='relu'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=9,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history=model.fit(trainSeq,trainCate,batch_size=256,epochs=40,validation_split=0.2)
model.save("TextCNNFENCI2")


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model_accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Valid'],loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model_loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Valid'],loc='upper left')
plt.show()
