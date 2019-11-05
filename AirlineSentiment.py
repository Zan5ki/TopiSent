if __name__ == '__main__':

    import re
    import pandas as pd
    import string
    import gensim
    import nltk
    from gensim.utils import simple_preprocess
    from gensim.parsing.preprocessing import STOPWORDS
    from nltk.stem import WordNetLemmatizer, SnowballStemmer
    from nltk.stem.porter import *
    import numpy as np
    np.random.seed(2018)

    stemmer = PorterStemmer()
    pd.set_option('display.max_columns', 1000)
    df = pd.read_csv("C:/Users/crozanski/Documents/CMKE136/Tweets.csv", low_memory=False)


    def preprocess_tweet(text):

        # Check characters to see if they are in punctuation
        nopunc = [char for char in text if char not in string.punctuation]
        # Join the characters again to form the string.
        nopunc = ''.join(nopunc)
        # convert text to lower-case
        nopunc = nopunc.lower()
        # remove URLs
        nopunc = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', nopunc)
        nopunc = re.sub(r'http\S+', '', nopunc)
        # remove usernames
        nopunc = re.sub('@[^\s]+', '', nopunc)
        # remove the # in #hashtag
        nopunc = re.sub(r'#([^\s]+)', r'\1', nopunc)
        return nopunc


    def lemmatize_stemming(text):
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


    def preprocess(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(lemmatize_stemming(token))
        return result


    df['text'] = df['text'].apply(preprocess_tweet)
    preprocessed_docs = df['text'].map(preprocess)

    dictionary = gensim.corpora.Dictionary(preprocessed_docs)

    # preview dictionary
    count = 0
    for k, v in dictionary.iteritems():
        print(k, v)
        count += 1
        if count > 20:
            break

    # filter extremes (no terms in: less than 15 total docs, over half of all docs) / keep top 100000 terms
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    # create bag of words dictionary for each doc (identify words in each doc and how many times each appears)
    bow_corpus = [dictionary.doc2bow(doc) for doc in preprocessed_docs]

    # preview bag of words for sample record
    bow_doc_4310 = bow_corpus[4310]
    for i in range(len(bow_doc_4310)):
        print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0],
                                                         dictionary[bow_doc_4310[i][0]],
                                                         bow_doc_4310[i][1]))

    # train LDA model using gensim.models.LdaMulticore and save it to ‘lda_model’
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
    # preview topics
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))
