from numpy.lib import average
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,learning_curve,GridSearchCV
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn import preprocessing, linear_model, naive_bayes, metrics
from sklearn import decomposition, ensemble
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,f1_score


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
df['Clause'] = df['Clause'].str.replace('\d', '',regex=True)
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
    # encoder = preprocessing.LabelEncoder()
    # Y_train = encoder.fit_transform(Y_train)
    # Y_test = encoder.fit_transform(Y_test)

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
    #plt.savefig('{}.png'.format(estimator))


def svm(tag,parameters):
    xtrain_tfidf_ngram,xvalid_tfidf_ngram,Y_train,Y_test = data(tag)
    model = SVC(probability=True)
    svm_model = GridSearchCV(model, parameters)
    svm_model.fit(xtrain_tfidf_ngram,Y_train)

    prediction = svm_model.predict(xvalid_tfidf_ngram)
    acc = metrics.accuracy_score(prediction,Y_test)
    f1_scr = f1_score(Y_test,prediction,average="weighted")
    #learningCurve(estimator=svm_model,X_train=xtrain_tfidf_ngram,Y_train=Y_train)
    return {"model":svm_model,"accuracy":acc,"f1_Score":f1_scr}

    

def naive_bayes_classifier(tag):
    X_train,X_test,Y_train,Y_test = train_test_split(df.Clause,df[tag],
    test_size=0.2,random_state = 123)

    # encoder = preprocessing.LabelEncoder()
    # Y_train = encoder.fit_transform(Y_train)
    # Y_test = encoder.fit_transform(Y_test)
    # print(np.unique(Y_test))
    #Extracting features from text files

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)

    # TF-IDF

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    prior_tag1 = [0.676,0.198,0.077,0.049]
    #prior_tag2 = [0.54,0.21,0.09,0.08,0.06,0.02]
    #prior_tag3 = [0.26,0.23,0.12,0.11,0.09,0.09,0.07,0.03]
    #clf = MultinomialNB().fit(X_train_tfidf, Y_train)
    # prior = None
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB(class_prior = prior_tag1))])
    text_clf = text_clf.fit(X_train,Y_train)

    # Performance of NB Classifier

    predicted = text_clf.predict(X_test)
    acc = np.mean(predicted == Y_test)
    f1_scr = f1_score(Y_test,predicted,average="weighted")
    #learningCurve(estimator=text_clf,X_train=X_train,Y_train=Y_train)
    return {"model":text_clf,"accuracy":acc,"f1_Score":f1_scr}


def random_forest(tag,parameters):
    xtrain_tfidf_ngram,xvalid_tfidf_ngram,Y_train,Y_test = data(tag)
    model = ensemble.RandomForestClassifier()
    rf = GridSearchCV(model,parameters)
    rf.fit(xtrain_tfidf_ngram, Y_train)

    predicted = rf.predict(xvalid_tfidf_ngram)
    print(rf.best_params_)
    f1_scr = f1_score(Y_test,predicted,average="weighted")
    acc = metrics.accuracy_score(Y_test,predicted)
    #learningCurve(estimator=rf,X_train=xtrain_tfidf_ngram,Y_train=Y_train)
    return {"model":rf,"accuracy":acc,"f1_Score":f1_scr}

def linearmodel(tag):
    xtrain_tfidf_ngram,xvalid_tfidf_ngram,Y_train,Y_test = data(tag)
    # learning rate
    model = linear_model.LogisticRegression()
    rf = model.fit(xtrain_tfidf_ngram, Y_train)

    predicted = rf.predict(xvalid_tfidf_ngram)
    f1_scr = f1_score(Y_test,predicted,average="weighted")
    acc = metrics.accuracy_score(Y_test,predicted)
    #learningCurve(estimator=rf,X_train=xtrain_tfidf_ngram,Y_train=Y_train)
    return {"model":rf,"accuracy":acc,"f1_Score":f1_scr}

def gradientboosting(tag,parameters):
    xtrain_tfidf_ngram,xvalid_tfidf_ngram,Y_train,Y_test = data(tag)
    # learning rate
    model = ensemble.GradientBoostingClassifier()
    gb = GridSearchCV(model,parameters)
    gb.fit(xtrain_tfidf_ngram, Y_train)

    predicted = gb.predict(xvalid_tfidf_ngram)
    f1_scr = f1_score(Y_test,predicted,average="weighted")
    acc = metrics.accuracy_score(Y_test,predicted)
    print(gb.best_params_)
    #learningCurve(estimator=gb,X_train=xtrain_tfidf_ngram,Y_train=Y_train)
    return {"model":gb,"accuracy":acc,"f1_Score":f1_scr}

def decision_tree(tag):
    xtrain_tfidf_ngram,xvalid_tfidf_ngram,Y_train,Y_test = data(tag)
    # learning rate
    model = DecisionTreeClassifier(max_depth=10)
    decision_tree = model.fit(xtrain_tfidf_ngram, Y_train)

    predicted = decision_tree.predict(xvalid_tfidf_ngram)
    acc = metrics.accuracy_score(Y_test,predicted)
    f1_scr = f1_score(Y_test,predicted,average="weighted")
    #learningCurve(estimator=decision_tree,X_train=xtrain_tfidf_ngram,Y_train=Y_train)
    return {"model":decision_tree,"accuracy":acc,"f1_Score":f1_scr}

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


# param_rf = {
#     'n_estimators': [100,300,500,600,700,800],
#     "max_depth":[20,22,26,28,30,32,34,36]
# }
param_rf = {'max_depth': [32], 'n_estimators': [600]}

# param_boosting = {"n_estimators":[300,400],"max_depth":[10,15]}
param_boosting = {"n_estimators":[300],"max_depth":[10]}

# print("Naive Bayes Tag3: ",naive_bayes_classifier("Tag3"))
# print("SVM Tag3: ",svm("Tag1",{'kernel':('linear', 'rbf')}))
# print("Random Forest Tag3: ",random_forest("Tag3",param_rf))
# print("Linear Model Tag3: ",linearmodel("Tag3"))
# print("Gradient Boosting Tag3: ",gradientboosting("Tag3",param_boosting))
# print("Decision Tree Tag3: ",decision_tree("Tag3"))

#topic_modeling("Tag1")


def voting_clasifier(tag):
    clf1 = linearmodel(tag)['model']
    clf2 = svm(tag,{'kernel':('linear', 'rbf')})['model']
    clf3 = random_forest(tag,param_rf)['model']
   
    eclf3 = ensemble.VotingClassifier(estimators=[('lr', clf1), ('svm', clf2),
    ('rf',clf3)],voting='soft')#, weights=[2,1,1],flatten_transform=True)

    xtrain_tfidf_ngram,xvalid_tfidf_ngram,Y_train,Y_test = data(tag)

    eclf3 = eclf3.fit(xtrain_tfidf_ngram,Y_train)

    pred = eclf3.predict(xvalid_tfidf_ngram)
    acc = metrics.accuracy_score(Y_test,pred)
    print("VotingClassifier Accuracy: ",acc)
    return eclf3

def classify(par,tag):
    model = voting_clasifier(tag)
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',max_features=5000)
    tfidf_vect_ngram.fit(df["Clause"])
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(par)
    pred = model.predict(xtrain_tfidf_ngram)
    return pred

