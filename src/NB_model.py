from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import process_text as pt
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def nb_model(news_train, news_test, text_train, text_test):

    pipeline1 = Pipeline([
        ('bow', CountVectorizer(analyzer=pt.process_text)),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])

    pipeline1.fit(news_train, text_train)
    predictions_NB = pipeline1.predict(news_test)

    print('NB - test')
    print(classification_report(predictions_NB, text_test))

    cm = confusion_matrix(text_test, predictions_NB)
    class_label = [0, 1]
    df_cm = pd.DataFrame(cm, index=class_label, columns=class_label)
    sns.heatmap(df_cm, annot=True, fmt='d',cmap="crest")
    plt.title("Confusion Matrix - Naive Bayes")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()