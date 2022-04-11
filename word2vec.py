from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import nltk
nltk.download('brown')
from nltk.corpus import brown

#training model with Brown corpus
train_set = brown.sents()
model = gensim.models.Word2Vec(train_set)
#saving model
model.save('brown.embedding')
brownModel = gensim.models.Word2Vec.load('brown.embedding')

#checking for some similarities for report (3 similar, 3 different)
print(brownModel.wv.similarity('university','school'))
print(brownModel.wv.similarity('kid','child'))
print(brownModel.wv.similarity('clock','watch'))

print(brownModel.wv.similarity('apple','color'))
print(brownModel.wv.similarity('airplane','apple'))
print(brownModel.wv.similarity('music','water'))