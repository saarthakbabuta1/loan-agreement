import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,learning_curve,GridSearchCV
from nltk.stem.snowball import SnowballStemmer
import pandas as pd


#nltk.download('stopwords')
#nltk.download('punkt')

def document_similarity(text1,text2,method="cosine"):
    stopwords_en = stopwords.words("english")
    def pre_processing(raw):
        wordlist = nltk.word_tokenize(raw)
        text = [w.lower() for w in wordlist if w not in stopwords_en]
        return text

    text1 = pre_processing(text1)
    text2 = pre_processing(text2)

    word_set = set(text1).union(set(text2))

    freqd_text1 = FreqDist(text1)
    text1_count_dict = dict.fromkeys(word_set,0)
    text1_tf_dict = dict.fromkeys(word_set,0)
    text1_length = text1_length = len(freqd_text1)
    for word in text1:
        text1_count_dict[word] = freqd_text1[word]
        text1_tf_dict[word] = freqd_text1[word]/text1_length

    freqd_text2 = FreqDist(text2)
    text2_tf_dict = dict.fromkeys(word_set,0)
    text2_length = text2_length = len(freqd_text2)
    text2_count_dict = dict.fromkeys(word_set,0)
    for word in text1:
        text2_count_dict[word] = freqd_text2[word]
        text2_tf_dict[word] = freqd_text2[word]/text2_length

    text12_idf_dict = dict.fromkeys(word_set,0)
    text12_length = 2 
    for word in text12_idf_dict.keys():
        if word in text1:
            text12_idf_dict[word] += 1
        if word in text2:
            text12_idf_dict[word] += 1

    for word,val in text12_idf_dict.items():
        text12_idf_dict[word] = 1 + math.log(text12_length/float(val))

    text1_df_idf_dict = dict.fromkeys(word_set,0)
    for word in text1:
        text1_df_idf_dict[word] = (text1_tf_dict[word] * text12_idf_dict[word])

    text2_df_idf_dict = dict.fromkeys(word_set,0)
    for word in text2:
        text2_df_idf_dict[word] = (text2_tf_dict[word] * text12_idf_dict[word])
    
    similarity = None
    if method == "cosine":
        v1 = list(text1_df_idf_dict.values())
        v2 = list(text2_df_idf_dict.values())

        similarity = 1 - nltk.cluster.cosine_distance(v1,v2)
    return similarity

df = pd.read_csv("mapping_nb.csv")

stop = stopwords.words('english')
stop.extend(['a','an','the','to'])

df['Clause'] = df['Clause'].str.lower()
df['Clause'] = df['Clause'].str.replace('\t','')
df['Clause'] = df['Clause'].str.replace('\n',' ')
df['Clause'] = df['Clause'].str.replace(r"\(.*\)","",regex=True)
df['Clause'] = df['Clause'].str.replace('\d.\d.\d.','',regex=True)
df['Clause'] = df['Clause'].str.replace('\d', '',regex=True)
df['Clause'] = df['Clause'].str.split(' ').apply(lambda x: ' '.join(k for k in x if k not in stop))
df['Clause'] = df['Clause'].str.replace('[^\w\s]','',regex=True)
df['Clause'] = df['Clause'].str.strip()


def similarity_file(df,Tag):
    data = []
    for i,row in df.iterrows():
        #print(row['Clause'])
        print(i)
        for i1,row1 in df.iterrows():
            try:
                data.append([row['Clause'],row1['Clause'],row[Tag],row1[Tag],document_similarity(row['Clause'],row1['Clause'])])
            except Exception as e:
                data.append([row['Clause'],row1['Clause'],row[Tag],row1[Tag],None])

    document_similarity_df = pd.DataFrame(data)
    document_similarity_df.columns = ['Clause1','Clause2','Tag1','Tag2','Text_Similarity']
    group_data = document_similarity_df.groupby(['Clause1','Tag1','Tag2'],as_index=False).Text_Similarity.mean()

    document_similarity_df.to_csv("doc.csv")
    group_data.to_csv("text_similarity.csv")

    return "Check text_similarity.csv"

similarity_file(df,"Tag3")
