if __name__ == '__main__':
    import pandas as pd
    import re
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyser = SentimentIntensityAnalyzer()

    df = pd.read_csv("C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/DomTopics.csv", sep=",")
    df_test = pd.read_csv("C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/DomTopicsTest.csv", sep=",")
    df_Doms = pd.read_csv("C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/DomTopics.csv", sep=",")
    df_Doms_test = pd.read_csv("C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/DomTopicsTest.csv", sep=",")
    df['text'] = df['text'].astype(str)
    df_test['text'] = df_test['text'].astype(str)
    df['text'] = df['text'].astype(str)
    mask = df['text'].str.len() > 299
    df = df.loc[mask]
    df_test['text'] = df_test['text'].astype(str)
    mask_test = df_test['text'].str.len() > 299
    df_test = df_test.loc[mask_test]


    # preprocess function
    def fixtexts(sentence):
        for sents in sentence:
            sents = re.sub(r'[^A-Za-z0-9:\-)\\(\[\]\'\s]+', '', sents)
            sents = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', sents)
            sents = re.sub(r'http\S+', '', sents)
            sents = re.sub(r'\d+', '', sents)
            sents = ' '.join(sents.split())
            yield (sents)


    # VADER function
    def vader(sentences):
        scores = []
        for sentence in sentences:
            score = analyser.polarity_scores(sentence)
            scores.append(score)
        return scores


    # process text
    fixed = df['text'].values.tolist()
    fixed = list(fixtexts(fixed))
    fixed_df = pd.DataFrame(fixed)
    fixed_df = fixed_df.reset_index()
    fixed_df.to_csv('C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/orig.csv', sep=",")
    df_VADERscores = pd.DataFrame(vader(fixed))
    df_VADERscores.to_csv('C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/VADERscores.csv', sep=",")
    df_VADERaverages = df_VADERscores.mean()
    df_VADERaverages.to_csv('C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/VADERaverages.csv', sep=",")
    fixed_test = df_test['text'].values.tolist()
    fixed_test = list(fixtexts(fixed_test))
    fixed_test_df = pd.DataFrame(fixed_test)
    fixed_test_df = fixed_test_df.reset_index()
    df_VADERscores_test = pd.DataFrame(vader(fixed_test))
    df_VADERscores_test.to_csv('C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/VADERscores_test.csv', sep=",")
    df_VADERaverages_test = df_VADERscores_test.mean()
    df_VADERaverages_test.to_csv('C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/VADERaverages_test.csv', sep=",")

    # create tables
    df_finished = pd.DataFrame()
    df_finished = df_finished.reset_index()
    df = df.reset_index()
    df_finished['text'] = df['text']
    df_VADERscores = df_VADERscores.reset_index()
    df_finished['compound'] = df_VADERscores['compound']
    df_finished['neg'] = df_VADERscores['neg']
    df_finished['neu'] = df_VADERscores['neu']
    df_finished['pos'] = df_VADERscores['pos']
    df_Doms = df_Doms.reset_index()
    df_finished['Document_No'] = df_Doms['Document_No']
    df_finished['Dominant_Topic'] = df_Doms['Dominant_Topic']
    df_finished['Topic_Perc_Contrib'] = df_Doms['Topic_Perc_Contrib']
    df_finished['Keywords'] = df_Doms['Keywords']
    df_finished['Rep_Text'] = df_Doms['Text']

    df_finished.to_csv('C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/WikisVADERscoresDomTopics.csv', sep=",")

    df_finished_test = pd.DataFrame()
    df_finished_test = df_finished_test.reset_index()
    df_test = df_test.reset_index()
    df_finished_test['text'] = df_test['text']
    df_VADERscores_test = df_VADERscores_test.reset_index()
    df_finished_test['compound'] = df_VADERscores_test['compound']
    df_finished_test['neg'] = df_VADERscores_test['neg']
    df_finished_test['neu'] = df_VADERscores_test['neu']
    df_finished_test['pos'] = df_VADERscores_test['pos']
    df_Doms_test = df_Doms_test.reset_index()
    df_finished_test['Document_No'] = df_Doms_test['Document_No']
    df_finished_test['Dominant_Topic'] = df_Doms_test['Dominant_Topic']
    df_finished_test['Topic_Perc_Contrib'] = df_Doms_test['Topic_Perc_Contrib']
    df_finished_test['Keywords'] = df_Doms_test['Keywords']
    df_finished_test['Rep_Text'] = df_Doms_test['Text']

    df_finished_test.to_csv(
        'C:/Users/crozanski/Documents/CMKE136/wikisent2.txt/WikisVADERscoresDomTopics_test.csv', sep=",")
