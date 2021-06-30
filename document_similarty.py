import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import math

nltk.download('stopwords')
nltk.download('punkt')

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
    

    # print("Text1",text1_count_dict)
    # print("Text1-tf",text1_tf_dict)
    

    # print("Text2",text2_count_dict)
    # print("Text2-tf",text2_tf_dict)

    # print("IDF",text12_idf_dict)

    # print("Text1 DF IDF",text1_df_idf_dict)
    # print("Text2 DF IDF",text2_df_idf_dict)

    similarity = None
    if method == "cosine":
        v1 = list(text1_df_idf_dict.values())
        v2 = list(text2_df_idf_dict.values())

        similarity = 1 - nltk.cluster.cosine_distance(v1,v2)
    return similarity



print(document_similarity("Hi, I am Saarthak","Hello, I am Tanvi"))