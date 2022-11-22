# Python program to train NB, SVM and DT models
import pandas as pd
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

truenews = pd.read_csv('datasets/True.csv')[:100]
fakenews = pd.read_csv('datasets/Fake.csv')[:100]

truenews['True/Fake']='True'
fakenews['True/Fake']='Fake'

# Combine the 2 DataFrames into a single data frame
news = pd.concat([truenews, fakenews])
news["Article"] = news["title"] + news["text"]


# Data Cleaning
def process_text(s):

    nopunc = [char for char in s if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    #
    clean_string = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_string


# Tokenization - tokens are words that we actually want
news['Clean Text'] = news['Article'].apply(process_text)

bow_transformer = CountVectorizer(analyzer=process_text).fit(news['Clean Text'])


# Bag-of-Words (bow)
news_bow = bow_transformer.transform(news['Clean Text'])

print('Shape of Sparse Matrix: ', news_bow.shape)
print('Amount of Non-Zero occurences: ', news_bow.nnz)


sparsity = (100.0 * news_bow.nnz / (news_bow.shape[0] * news_bow.shape[1]))
print('Sparisty: {}'.format(round(sparsity)))



# TF-IDF


tfidf_transformer = TfidfTransformer().fit(news_bow)
news_tfidf = tfidf_transformer.transform(news_bow)



# Training Naive Bayes Model

NB = MultinomialNB().fit(news_tfidf, news['True/Fake'])

# Model Evaluation
predictions_NB = NB.predict(news_tfidf)


SVM = SVC(C=1.0, kernel='linear', gamma='auto').fit(news_tfidf, news['True/Fake'])
predictions_SVM = SVM.predict(news_tfidf)

print('NB Predictions')
print(classification_report(news['True/Fake'], predictions_NB))

# SVM predictions
print('SVM Predictions')
print(classification_report(news['True/Fake'], predictions_SVM))


news_train, news_test, text_train, text_test = train_test_split(news['Article'], news['True/Fake'], test_size=0.3)


#  Data Pipelines
pipeline1 = Pipeline([
    ('bow', CountVectorizer(analyzer=process_text)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

pipeline2 = Pipeline([
    ('bow', CountVectorizer(analyzer=process_text)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', SVC()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

# Fitting and predictions
pipeline1.fit(news_train, text_train)
predictions_NB = pipeline1.predict(news_test)

pipeline2.fit(news_train, text_train)
predictions_SVM = pipeline2.predict(news_test)

# Naive Bayes
print('NB - test')
print(classification_report(predictions_NB, text_test))

# SVM
print('SVM - test')
print(classification_report(predictions_SVM, text_test))
