import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split,learning_curve
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn import preprocessing, linear_model, naive_bayes, metrics
from sklearn import decomposition, ensemble
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv("mapping_nb.csv")

stop = stopwords.words('english')
stop.extend(['a','an','the','to'])

#nltk.download('wordnet')
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])

df['Clause'] = df.Clause.apply(lemmatize_text)

stemmer = SnowballStemmer("english")

df['Clause'].apply(lambda x: ' '.join([stemmer.stem(y) for y in x]))

df['Clause'] = df['Clause'].str.lower()
df['Clause'] = df['Clause'].str.replace('\t','')
df['Clause'] = df['Clause'].str.replace('\n',' ')
df['Clause'] = df['Clause'].str.replace(r"\(.*\)","",regex=True)
df['Clause'] = df['Clause'].str.replace('\d.\d.\d.','',regex=True)
df['Clause'] = df['Clause'].str.replace('\d+', '',regex=True)
df['Clause'] = df['Clause'].str.split(' ').apply(lambda x: ' '.join(k for k in x if k not in stop))
df['Clause'] = df['Clause'].str.strip()


def confusionMatrix(Y_test,prediction):
    unique_label = np.unique(Y_test)
    cmtx = pd.DataFrame(
    confusion_matrix(prediction, Y_test, labels=unique_label), 
    index=['true:{:}'.format(x) for x in unique_label], 
    columns=['pred:{:}'.format(x) for x in unique_label])
    return cmtx

def data(tag):
    X_train,X_test,Y_train,Y_test = train_test_split(df.Clause,df[tag],
    test_size=0.2,random_state = 1234)   

    # label encode the target variable 
    encoder = preprocessing.LabelEncoder()
    Y_train = encoder.fit_transform(Y_train)
    Y_test = encoder.fit_transform(Y_test)

    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',max_features=5000)
    tfidf_vect_ngram.fit(df["Clause"])
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(X_train)
    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(X_test)

    return xtrain_tfidf_ngram,xvalid_tfidf_ngram,Y_train,Y_test

def learningCurve(estimator,X_train,Y_train):
    train_sizes, train_scores, test_scores = learning_curve(estimator=estimator, X=X_train, y=Y_train,
                                                       cv=10, train_sizes=np.linspace(0.1, 1.0, 10),
                                                     n_jobs=1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    #
    # Plot the learning curve
    #
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.title('Learning Curve')
    plt.xlabel('Training Data Size')
    plt.ylabel('Model accuracy')
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()


def svm(tag,kernel = "linear"):
    xtrain_tfidf_ngram,xvalid_tfidf_ngram,Y_train,Y_test = data(tag)
    model = SVC(kernel=kernel)
    svm_model = model.fit(xtrain_tfidf_ngram,Y_train)

    prediction = svm_model.predict(xvalid_tfidf_ngram)
    acc = metrics.accuracy_score(prediction,Y_test)

    print(confusionMatrix(Y_test,prediction))
    learningCurve(estimator=svm_model,X_train=xtrain_tfidf_ngram,Y_train=Y_train)
    return acc

    

def naive_bayes_classifier(tag):
    X_train,X_test,Y_train,Y_test = train_test_split(df.Clause,df[tag],
    test_size=0.2,random_state = 123)

    encoder = preprocessing.LabelEncoder()
    Y_train = encoder.fit_transform(Y_train)
    Y_test = encoder.fit_transform(Y_test)
    print(np.unique(Y_test))
    #Extracting features from text files

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)

    # TF-IDF

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    prior_tag1 = [0.676,0.198,0.077,0.049]
    #clf = MultinomialNB().fit(X_train_tfidf, Y_train)
    # prior = None
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB(class_prior = prior_tag1))])
    text_clf = text_clf.fit(X_train,Y_train)

    # Performance of NB Classifier

    predicted = text_clf.predict(X_test)
    print(confusionMatrix(Y_test,predicted))
    learningCurve(estimator=text_clf,X_train=X_train,Y_train=Y_train)
    return np.mean(predicted == Y_test)


def random_forest(tag,n_estimators=10,max_features = 500,max_depth=10):
    xtrain_tfidf_ngram,xvalid_tfidf_ngram,Y_train,Y_test = data(tag)
    model = ensemble.RandomForestClassifier(n_estimators = n_estimators,max_features = max_features,
    max_depth=max_depth)
    rf = model.fit(xtrain_tfidf_ngram, Y_train)

    predicted = rf.predict(xvalid_tfidf_ngram)
    print(confusionMatrix(Y_test,predicted))
    learningCurve(estimator=rf,X_train=xtrain_tfidf_ngram,Y_train=Y_train)
    return metrics.accuracy_score(predicted,Y_test)

def linearmodel(tag):
    xtrain_tfidf_ngram,xvalid_tfidf_ngram,Y_train,Y_test = data(tag)
    # learning rate
    model = linear_model.LogisticRegression()
    rf = model.fit(xtrain_tfidf_ngram, Y_train)

    predicted = rf.predict(xvalid_tfidf_ngram)
    print(confusionMatrix(Y_test,predicted))
    learningCurve(estimator=rf,X_train=xtrain_tfidf_ngram,Y_train=Y_train)
    return metrics.accuracy_score(predicted,Y_test)

def gradientboosting(tag):
    xtrain_tfidf_ngram,xvalid_tfidf_ngram,Y_train,Y_test = data(tag)
    # learning rate
    model = ensemble.GradientBoostingClassifier(n_estimators=100,max_depth=10)
    gb = model.fit(xtrain_tfidf_ngram, Y_train)

    predicted = gb.predict(xvalid_tfidf_ngram)
    print(confusionMatrix(Y_test,predicted))
    learningCurve(estimator=gb,X_train=xtrain_tfidf_ngram,Y_train=Y_train)
    return metrics.accuracy_score(predicted,Y_test)

def decision_tree(tag):
    xtrain_tfidf_ngram,xvalid_tfidf_ngram,Y_train,Y_test = data(tag)
    # learning rate
    model = DecisionTreeClassifier(max_depth=10)
    decision_tree = model.fit(xtrain_tfidf_ngram, Y_train)

    predicted = decision_tree.predict(xvalid_tfidf_ngram)
    print(confusionMatrix(Y_test,predicted))
    learningCurve(estimator=decision_tree,X_train=xtrain_tfidf_ngram,Y_train=Y_train)
    return metrics.accuracy_score(predicted,Y_test)

def topic_modeling(tag):
    X_train,X_test,Y_train,Y_test = train_test_split(df.Clause,df[tag],
    test_size=0.2,random_state = 123)

    # Create a count vectorized Object
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(df['Clause'])

# transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(X_train)
    xvalid_count =  count_vect.transform(X_test)
    # train a LDA Model
    lda_model = decomposition.LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)
    X_topics = lda_model.fit_transform(xtrain_count)
    topic_word = lda_model.components_ 
    vocab = count_vect.get_feature_names()
    print(vocab)
    # view the topic models
    n_top_words = 5
    topic_summaries = []
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        topic_summaries.append(' '.join(topic_words))
    print(topic_summaries)


print("Naive Bayes Tag1: ",naive_bayes_classifier("Tag1"))
print("SVM Tag1: ",svm("Tag1"))
print("Random Forest Tag1: ",random_forest("Tag1",n_estimators=30,max_features=500,max_depth=25))
print("Linear Model: ",linearmodel("Tag2"))
print("Gradient Boosting: ",gradientboosting("Tag1"))
print("Decision Tree: ",decision_tree("Tag1"))

#topic_modeling("Tag1")

