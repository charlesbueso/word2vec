from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import nltk
nltk.download('brown')
nltk.download('word2vec_sample')
from nltk.corpus import brown
from nltk.data import find
import gensim.downloader as api

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

#loading pre-trained google model
#pruned to only include 44k most common words
word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
googleModel = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)

#evaluating with word analogies to test for qualitativeness and overall accuracy
evalScore, listOfSections = googleModel.wv.evaluate_word_analogies(datapath('questions-words.txt'))

#printing results for each category.
for section in listOfSections:
    print(str(len(section['correct'])) + ' were correct, and incorrect: ' + str(len(section['incorrect'])))

#Calculating Pearson correlation, Spearman correlation, and ratio of pairs of unknown words
googleModel.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))

# Check the "most similar words", using the default "cosine similarity" measure.
# woman + king - man
result = googleModel.most_similar(positive=['woman', 'king'], negative=['man'])
most_similar_key, similarity = result[0]  # look at the first match
print(f"{most_similar_key}: {similarity:.4f}")
# ice + liquid - water
result = googleModel.most_similar(positive=['ice', 'liquid'], negative=['water'])
most_similar_key, similarity = result[0]  # look at the first match
print(f"{most_similar_key}: {similarity:.4f}")
# girl + man - boy
result = googleModel.most_similar(positive=['girl', 'man'], negative=['boy'])
most_similar_key, similarity = result[0]  # look at the first match
print(f"{most_similar_key}: {similarity:.4f}")
# green + sky - grass
result = googleModel.most_similar(positive=['green', 'sky'], negative=['grass'])
most_similar_key, similarity = result[0]  # look at the first match
print(f"{most_similar_key}: {similarity:.4f}")
# puppy + cat - dog
result = googleModel.most_similar(positive=['puppy', 'cat'], negative=['dog'])
most_similar_key, similarity = result[0]  # look at the first match
print(f"{most_similar_key}: {similarity:.4f}")