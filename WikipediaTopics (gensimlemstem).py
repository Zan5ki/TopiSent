if __name__ == '__main__':

    import pandas as pd
    import gensim
    import time
    import pickle
    import numpy as np
    import os
    import pyLDAvis.gensim
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns
    from collections import Counter
    from nltk.stem.porter import *
    from nltk import WordNetLemmatizer
    from nltk.corpus import stopwords
    from gensim import corpora, models
    from gensim.models import CoherenceModel
    from sklearn.manifold import TSNE
    from sklearn.model_selection import train_test_split
    from bokeh.plotting import figure, output_file, show
    from matplotlib import pyplot as plt
    from wordcloud import WordCloud

    startpreprocesstime = time.time()

    np.random.seed(2018)
    os.environ['MALLET_HOME'] = 'C:/Users/crozanski/Documents/CMKE136/mallet-2.0.8/'
    mallet_path = 'C:/Users/crozanski/Documents/CMKE136/mallet-2.0.8/bin/mallet.bat'
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    pd.set_option('display.max_columns', 1000)
    lines = open("C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/wikisent2.txt").read().splitlines()
    df = pd.DataFrame(lines)
    df[0] = df[0].astype(str)
    mask = df[0].str.len() > 299
    df = df.loc[mask]
    df.columns = ['text']
    df, testset = train_test_split(df, test_size=0.014)


    def preprocess(sentences):
        for sent in sentences:
            # remove punctuation and special characters
            sent = re.sub(r'[^A-Za-z0-9\s]+', '', sent)
            sent = ''.join(sent)
            sent = sent.lower()
            # remove URLs
            sent = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', sent)
            sent = re.sub(r'http\S+', '', sent)
            # remove numbers
            sent = re.sub(r'\d+', '', sent)
            yield (sent)


    data = df['text'].values.tolist()
    datatest = testset['text'].values.tolist()
    preprocessed_docs = list(preprocess(data))
    preprocessed_docs_test = list(preprocess(datatest))
    preprocessed_docs = pd.DataFrame(preprocessed_docs)
    preprocessed_docs_test = pd.DataFrame(preprocessed_docs_test)
    preprocessed_docs.columns = ['text']
    preprocessed_docs_test.columns = ['text']
    preprocessed_docs.to_csv(
        'C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/textpreprocess.csv', sep=',', encoding='utf-8', index=False)

    endpreprocesstime = time.time()
    processtime = endpreprocesstime - startpreprocesstime
    print(processtime)

    startstemlemtime = time.time()


    def lemstem(text):
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


    def lemstemtokenizestopwords(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(lemstem(token))
        return result


    preprocessed_docs = preprocessed_docs['text'].apply(lemstemtokenizestopwords)
    preprocessed_docs_test = preprocessed_docs_test['text'].apply(lemstemtokenizestopwords)
    preprocessed_docs = preprocessed_docs.tolist()
    preprocessed_docs_test = preprocessed_docs_test.tolist()
    print(type(preprocessed_docs))
    print(preprocessed_docs)

    endstemlemtime = time.time()
    stemlemtime = endstemlemtime - startstemlemtime
    print(stemlemtime)

    startmodeltime = time.time()

    diction = gensim.corpora.Dictionary(preprocessed_docs)

    # preview dictionary
    count = 0
    for k, v in diction.iteritems():
        print(k, v)
        count += 1
        if count > 50:
            break

    # filter extremes (no terms in: less than 15 total docs, over half of all docs) / keep top 100000 terms
    diction.filter_extremes(no_below=15, no_above=0.50, keep_n=100000)

    # create bag of words dictionary for each doc (identify words in each doc and how many times each appears)
    bow_corpus = [diction.doc2bow(doc) for doc in preprocessed_docs]
    pickle.dump(bow_corpus, open('gensim_corpus_corpus.pkl', 'wb'))
    diction.save('C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/gensim_dictionary.gensim')

    # preview bag of words for sample record
    bow_doc12345 = bow_corpus[12345]
    for i in range(len(bow_doc12345)):
        print("Word {} (\"{}\") appears {} time.".format(bow_doc12345[i][0],
                                                         diction[bow_doc12345[i][0]],
                                                         bow_doc12345[i][1]))


    def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        coherence_valuess = []
        model_lists = []
        for num_topics in range(start, limit, step):
            model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=diction)
            model_lists.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_valuess.append(coherencemodel.get_coherence())

        return model_lists, coherence_valuess


    # compute coherence scores
    model_list, coherence_values = compute_coherence_values(
        dictionary=diction, corpus=bow_corpus, texts=preprocessed_docs, start=2, limit=40, step=6)

    # generate coherence graph
    limit = 40
    start = 2
    step = 6
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend("coherence_values", loc='best')
    plt.savefig('C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/Ideal_Topic_Num.png')

    startmodeltime = time.time()

    # create model
    ldamalletBOW = gensim.models.wrappers.LdaMallet(
        mallet_path, corpus=bow_corpus, num_topics=10, id2word=diction, workers=2)

    # display topics
    for idx, topic in ldamalletBOW.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

    # print coherence score for model
    coherence_model_lda = CoherenceModel(
        model=ldamalletBOW, texts=preprocessed_docs, dictionary=diction, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    # generate model visualization
    malletBOWmodeltoldamodel = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldamalletBOW)
    malletBOWmodeltoldamodel.save('C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/gensim_model.gensim')
    BOWvis = pyLDAvis.gensim.prepare(malletBOWmodeltoldamodel, bow_corpus, diction)
    pyLDAvis.save_html(BOWvis, 'C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/model.html')

    endmodeltime = time.time()
    modeltime = endmodeltime - startmodeltime
    print(modeltime)

    # classify training set
    startmodelprocesstime = time.time()


    def format_topics_sentences(ldamodel=malletBOWmodeltoldamodel, corpus=bow_corpus, texts=data):
        # initiate output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for e, row_list in enumerate(ldamodel[corpus]):
            row = row_list[0] if ldamodel.per_word_topics else row_list
            # print(row)
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([words for words, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return sent_topics_df


    df_topic_sents_keywords = format_topics_sentences(
        ldamodel=malletBOWmodeltoldamodel, corpus=bow_corpus, texts=preprocessed_docs)
    df_topic_sents_keywords_test = format_topics_sentences(
        ldamodel=malletBOWmodeltoldamodel, corpus=bow_corpus, texts=preprocessed_docs_test)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic_test = df_topic_sents_keywords_test.reset_index()
    df = df.reset_index()
    testset = testset.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    df_dominant_topic_test.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    df_dominant_topic['text'] = df['text']
    df_dominant_topic_test['text'] = testset['text']
    df_dominant_topic.to_csv(
        'C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/DomTopics.csv', sep=',', encoding='utf-8', index=False)
    df_dominant_topic_test.to_csv(
        'C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/DomTopicsTest.csv', sep=',', encoding='utf-8', index=False)

    endmodelprocesstime = time.time()
    modelprocesstime = endmodelprocesstime - startmodelprocesstime
    print(modelprocesstime)

    # generate model results statistics and visualizations
    startstatsandvistime = time.time()

    sent_topics_sorteddf_mallet = pd.DataFrame()
    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                                 grp.sort_values(['Perc_Contribution'], ascending=False).head(1)],
                                                axis=0)

    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

    sent_topics_sorteddf_mallet.to_csv(
        'C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/BestSentences.csv', sep=',', encoding='utf-8', index=False)

    doc_lens = [len(d) for d in df_dominant_topic.Text]

    # plot corpus word count distribution
    plt.figure(figsize=(16, 7), dpi=160)
    plt.hist(doc_lens, bins=250, color='navy')
    plt.text(175, 4000, "Mean   : " + str(round(np.mean(doc_lens))))
    plt.text(175, 3500, "Median : " + str(round(np.median(doc_lens))))
    plt.text(175, 3000, "Stdev   : " + str(round(np.std(doc_lens))))
    plt.text(175, 2500, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
    plt.text(175, 2000, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))

    plt.gca().set(xlim=(0, 250), ylabel='Number of Documents', xlabel='Document Word Count')
    plt.tick_params(size=16)
    plt.xticks(np.linspace(0, 250, 26))
    plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
    plt.savefig('C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/WordCountDistribution.png')

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # fewer colors: 'TABLEAU_COLORS'

    fig, axes = plt.subplots(2, 5, figsize=(18, 15), dpi=160, sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]
        doc_lens1 = [len(q) for q in df_dominant_topic_sub.Text]
        ax.hist(doc_lens1, bins=300, color=cols[i])
        ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
        sns.kdeplot(doc_lens1, color="black", shade=False, ax=ax.twinx())
        ax.set(xlim=(0, 120), xlabel='Document Word Count')
        ax.set_ylabel('Number of Documents', color=cols[i])
        ax.set_title('Topic: ' + str(i), fontdict=dict(size=16, color=cols[i]))

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.xticks(np.linspace(0, 120, 7))
    fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=22)
    plt.savefig('C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/WordCountDistributionsByTopic.png')

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # fewer colors: 'mcolors.TABLEAU_COLORS'

    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    topics = malletBOWmodeltoldamodel.show_topics(formatted=False)

    fig, axes = plt.subplots(2, 5, figsize=(12, 12), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.savefig('C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/WordCloudsByTopic.png')

    topics = malletBOWmodeltoldamodel.show_topics(formatted=False)
    data_flat = [w for w_list in preprocessed_docs for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i, weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

    # plot word count and weights of topic Keywords
    fig, axes = plt.subplots(2, 5, figsize=(18, 12), sharey=True, dpi=160)
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.5, alpha=0.3,
               label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.2,
                    label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, 0.080)
        ax.set_ylim(0, 25000)
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id == i, 'word'], rotation=30, horizontalalignment='right')
        ax.legend(loc='upper left')
        ax_twin.legend(loc='upper right')

    fig.tight_layout(w_pad=2)
    fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)
    plt.savefig('C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/WordCountsByTopic.png')

    # t-SNE
    # get topic weights
    topic_weights = []
    for row_list in malletBOWmodeltoldamodel[bow_corpus]:
        tmp = np.zeros(10)
        for i, w in row_list:
            tmp[i] = w
        topic_weights.append(tmp)

    # array of topic weights
    arr = pd.DataFrame(topic_weights).fillna(0).values

    # dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    # tSNE dimension reduction
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)

    # plot the topic clusters using Bokeh
    output_file("C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/TSNEClustering.html")
    n_topics = 10
    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics),
                  plot_width=900, plot_height=700)
    plot.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=mycolors[topic_num])
    show(plot)

    endstatsandvistime = time.time()
    statsandvistime = endstatsandvistime - startstatsandvistime
    print(statsandvistime)

    runtimesandcoherence = pd.DataFrame({'processtime': [processtime],
                                         'stopsstemlemtime': [stemlemtime],
                                         'modeltime': [modeltime],
                                         'modelprocesstime': [modelprocesstime],
                                         'statsandvistime': [statsandvistime],
                                         'CoherenceScore': [coherence_lda]})

    runtimesandcoherence.to_csv('C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/runtimes.csv', sep=',')
