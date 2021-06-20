import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist

nltk.download('stopwords')
nltk.download('punkt')

def document_similarity(text1,text2):
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
    for word in text1:
        text1_count_dict[word] = freqd_text1[word]

    freqd_text2 = FreqDist(text2)
    text2_count_dict = dict.fromkeys(word_set,0)
    for word in text1:
        text2_count_dict[word] = freqd_text2[word]


    print("Text1",text1_count_dict)

    print("Text2",text2_count_dict)


document_similarity("Hi, I am Saarthak","Hello, I am Tanvi")