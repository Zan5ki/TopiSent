if __name__ == '__main__':

    import pandas as pd
    import string
    import gensim
    from gensim.utils import simple_preprocess
    from gensim.parsing.preprocessing import STOPWORDS
    from nltk.stem import WordNetLemmatizer
    from nltk.stem.porter import *
    import numpy as np
    np.random.seed(2018)
    import nltk
    nltk.download('wordnet')
    stemmer = PorterStemmer()
    pd.set_option('display.max_columns', 1000)
    df = pd.read_csv("C:/Users/crozanski/Documents/CMKE136/Tweets.csv", low_memory=False)


    def preprocess_tweet(text):

        # remove punctuation
        preprocessing = [char for char in text if char not in string.punctuation]
        # rejoin characters
        preprocessing = ''.join(preprocessing)
        # convert text to lowercase
        preprocessing = preprocessing.lower()
        # remove URLs
        preprocessing = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', preprocessing)
        preprocessing = re.sub(r'http\S+', '', preprocessing)
        # remove usernames
        preprocessing = re.sub('@[^\s]+', '', preprocessing)
        # remove #s
        preprocessing = re.sub(r'#([^\s]+)', r'\1', preprocessing)
        return preprocessing


    def lemstem(text):
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


    def lemstemtokenizestopwords(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(lemstem(token))
        return result


    df['text'] = df['text'].apply(preprocess_tweet)
    preprocessed_docs = df['text'].map(lemstemtokenizestopwords)

    dictionary = gensim.corpora.Dictionary(preprocessed_docs)

    # preview dictionary
    count = 0
    for k, v in dictionary.iteritems():
        print(k, v)
        count += 1
        if count > 50:
            break

    # filter extremes (no terms in: less than 15 total docs, over half of all docs) / keep top 100000 terms
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    # create bag of words dictionary for each doc (identify words in each doc and how many times each appears)
    bow_corpus = [dictionary.doc2bow(doc) for doc in preprocessed_docs]

    # preview bag of words for sample record
    bow_doc12345 = bow_corpus[12345]
    for i in range(len(bow_doc12345)):
        print("Word {} (\"{}\") appears {} time.".format(bow_doc12345[i][0],
                                                         dictionary[bow_doc12345[i][0]],
                                                         bow_doc12345[i][1]))

    # train LDA model using gensim.models.LdaMulticore and save it to ‘lda_model’
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=8, id2word=dictionary, passes=2, workers=2)
    # display topics
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))
