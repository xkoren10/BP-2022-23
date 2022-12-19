from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import process_text as pt
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def dt_model(news_tfidf,news):
    DT = DecisionTreeClassifier().fit(news_tfidf, news['True/Fake'])
    predictions_DT = DT.predict(news_tfidf)

    print('DT Predictions')
    print(classification_report(news['True/Fake'], predictions_DT))

    news_train, news_test, text_train, text_test = train_test_split(news['Article'], news['True/Fake'], test_size=0.3)

    pipeline3 = Pipeline([
        ('bow', CountVectorizer(analyzer=pt.process_text)),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', DecisionTreeClassifier()),  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])

    pipeline3.fit(news_train, text_train)
    predictions_DT = pipeline3.predict(news_test)

    print('DT - test')
    print(classification_report(predictions_DT, text_test))

    cm = confusion_matrix(text_test, predictions_DT)
    class_label = [0, 1]
    df_cm = pd.DataFrame(cm, index=class_label, columns=class_label)
    sns.heatmap(df_cm, annot=True, fmt='d',cmap="crest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()