import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer


import os
import glob
import xml.etree.ElementTree as ET

from gensim.models import Word2Vec

import numpy as np
import pickle

# was used for old is_similar function
# from sklearn.metrics.pairwise import cosine_similarity




sia = SentimentIntensityAnalyzer()

# download the tools
nltk.download('punkt')
nltk.download('vader_lexicon')

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')



old_debates_path = "Collection of Data/old debates"
new_debates_path = "Collection of Data/new debates"

# get stopwords
stop_words = set(stopwords.words('english'))




#-- section of code for building lexicon
# vocab and embeddings from https://nlp.stanford.edu/projects/histwords/, 'All English (1800s-1990s)'
# load vocab
with open("1800-vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# load embeddings
embeddings = np.load("1800-w.npy")

# tokenize debates, based on get_all_speach, returns tokens instead of string
def get_tokenized_debates(directory):
    tokenized_texts = []
    for file in glob.glob(os.path.join(directory, "*.xml")):
        try:
            tree = ET.parse(file)
            root = tree.getroot()
            
            texts = [elem.text for elem in root.findall(".//membercontribution") if elem.text and elem.text.strip()]
            if texts:
                for text in texts:
                    tokens = word_tokenize(text)
                    tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
                    tokenized_texts.append(tokens)

        except ET.ParseError:
            print(f"Error parsing {file}")
    return tokenized_texts

# tokenize old debates
tokenized_old_debates = get_tokenized_debates(old_debates_path)

# tokenize new debates
tokenized_new_debates = get_tokenized_debates(new_debates_path)

# train on old debates
old_debates_model = Word2Vec(
    sentences=tokenized_old_debates,
    vector_size=300,
    window=10, # slightly larger than usual
    min_count=2,
    workers=1,
    sg=1,
    seed=1
)

# train on new debates
new_debates_model = Word2Vec(
    sentences=tokenized_new_debates,
    vector_size=300,
    window=10, # slightly larger than usual
    min_count=2,
    workers=1,
    sg=1,
    seed=1
)

# word to vector dictionary
old_word_to_vector = {word: old_debates_model.wv[word] for word in old_debates_model.wv.index_to_key}

# word to vector dictionary
new_word_to_vector = {word: new_debates_model.wv[word] for word in new_debates_model.wv.index_to_key}

# word_to_vector = {word: embeddings[idx] for idx, word in enumerate(vocab)} - keeping in case needed again


# get VADER lexicon
vader_lexicon = sia.lexicon


# find 5 most similar words
def most_similar(word, topn=5):
    if word not in old_debates_model.wv and new_debates_model.wv:
        return []
    if word not in old_debates_model.wv:
        return new_debates_model.wv.most_similar(word, topn=topn)
    return old_debates_model.wv.most_similar(word, topn=topn)


updated_words = {}

# write
# produce new lexicon
def build_lexicon(word_to_vector_choice):
    # for each word 
    for word in vader_lexicon:
        print('word in lexicon: ' + word)
        # find word in embeddings dictionary (formerly histwords)
        if word in word_to_vector_choice:
            similar_words_in_lexicon = []
            print('word in word to vecctor: ' + word)

            # find related word
            similar_words = most_similar(word)
            # for each similar word
            for similar_word in similar_words:
                print('word in similar: ')
                print(similar_word)
                # if similar word in lexicon
                if similar_word[0] in vader_lexicon:
                    print('similar word in lexicon: ')
                    print(similar_word)
                    # add word to list of words
                    print(similar_word[1])
                    similar_words_in_lexicon.append(vader_lexicon[similar_word[0]])
            
            if similar_words_in_lexicon:
                # get average of related words
                average = sum(similar_words_in_lexicon) / len(similar_words_in_lexicon)
                print(word)
                print(average)
                updated_words[word] = average
            else:
                average = vader_lexicon[word]
                print(word)
                print(average)
                updated_words[word] = average
        else:
            average = vader_lexicon[word]
            print(word)
            print(average)
            updated_words[word] = average
    return 'success'

# prepare the new lexicon, though it will not yet be implemented
#-- end of lexicon editing section

# get every word spoken
def get_all_speech(directory):
    contributions = []
    for file in glob.glob(os.path.join(directory, "*.xml")):
        try:
            tree = ET.parse(file)
            root = tree.getroot()
            
            # extract membercontribution tags' contents
            texts = [elem.text for elem in root.findall(".//membercontribution") if elem.text and elem.text.strip()]
            # combine text
            if texts:
                contributions.append(" ".join(texts))

        except ET.ParseError:
            print(f"Error parsing {file}")
    
    return contributions




# preprocess debates
def preprocess_debates(debates):
    # preprocessed debates
    preprocessed_debates = []

    # process each debate
    for debate in debates:
        tokenized_debate = word_tokenize(debate)
        # stop word removal
        filtered_tokens = [word.lower() for word in tokenized_debate if word.isalnum() and word.lower() not in stop_words]
        # tagging
        pos_tags = pos_tag(filtered_tokens)
        print(pos_tags)
        preprocessed_debates.append(pos_tags)
    
    return preprocessed_debates

# analyse debate
def analyse_sentiment(debates):
    sentiments = []

    for debate in debates:
        # convert debate to a string
        debate_text = ' '.join([word for word, _ in debate])
        
        # get score
        sentiment_scores = sia.polarity_scores(debate_text)
        sentiments.append(sentiment_scores)
    return sentiments

# saving lexicon
def save_lexicon(lexicon, filepath):
    with open(filepath, 'w', encoding='utf-8') as new_file:
        for word, score in lexicon.items():
            new_file.write(f"{word}\t{score:.4f}\n")


# get debates
print('getting debates')
old_debates = get_all_speech(old_debates_path)
new_debates = get_all_speech(new_debates_path)


old_preprocessed_debates = preprocess_debates(old_debates)
new_preprocessed_debates = preprocess_debates(new_debates)
print('begin preprocessing')
print('old:')
print(old_preprocessed_debates)
print('new:')
print(new_preprocessed_debates)




# train and build a lexicon with old debates
print(build_lexicon(old_word_to_vector))
sia.lexicon.update(updated_words)
save_lexicon(sia.lexicon, "old_lexicon.txt")


old_debates_analysis = analyse_sentiment(old_preprocessed_debates)


# train and build a lexicon with new debates
print(build_lexicon(new_word_to_vector))
sia.lexicon.update(updated_words)
save_lexicon(sia.lexicon, "new_lexicon.txt")



new_debates_analysis = analyse_sentiment(new_preprocessed_debates)

print(old_debates_analysis)
print(new_debates_analysis)

# get average negativity
def is_ruder(analysis1, analysis2):

    index1 = 0
    index2 = 0

    average1 = 0
    average2 = 0

    for score in analysis1:
        average1 += score['neg']
        index1 += 1
    for score in analysis2:
        average2 += score['neg']
        index2 += 1
    average1 = average1/index1
    average2 = average2/index2

    print('averages:')
    print("Old Debates:")
    print(average1)
    print("New Debates:")
    print(average2)

    if(average1 < average2):
        return True
    else:
        return False

print('politicians now are more negative: ')
print(is_ruder(old_debates_analysis, new_debates_analysis))

