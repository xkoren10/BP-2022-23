from sklearn.metrics import classification_report
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import process_text as pt
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def sv_model(news_train, news_test, text_train, text_test):

    pipeline2 = Pipeline([
        ('bow', CountVectorizer(analyzer=pt.process_text)),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', SVC(C=1.0, kernel='linear', gamma='auto')),  # train w/ Support Vectors classifier
    ])

    pipeline2.fit(news_train, text_train)
    predictions_SVM = pipeline2.predict(news_test)

    print('SVM - test')
    print(classification_report(predictions_SVM, text_test))

    cm = confusion_matrix(text_test, predictions_SVM)
    class_label = [0, 1]
    df_cm = pd.DataFrame(cm, index=class_label, columns=class_label)
    sns.heatmap(df_cm, annot=True, fmt='d',cmap="crest")
    plt.title("Confusion Matrix - Support vectors")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()