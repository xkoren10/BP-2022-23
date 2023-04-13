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
import dill as pickle
from datetime import datetime
from lime.lime_text import LimeTextExplainer


def nb_model(news_train, news_test, text_train, text_test, saved):

    pipeline_nb = Pipeline([
        # strings to token integer counts
        ('bow', CountVectorizer(analyzer=pt.process_text)),
        # integer counts to weighted TF-IDF scores
        ('tfidf', TfidfTransformer()),
        # train on TF-IDF vectors w/ Naive Bayes classifier
        ('classifier', MultinomialNB()),
    ])

    pipeline_nb.fit(news_train, text_train)
    predictions_NB = pipeline_nb.predict(news_test)

    if saved == True:
        filename = 'saved_models/model_nb.pk'
        with open(filename, 'wb') as file:       # save model when training on new dataset
            pickle.dump(pipeline_nb, file)

    # Classification report
    print('NB - test')
    print(classification_report(predictions_NB, text_test))

    # Confusion matrix
    cm = confusion_matrix(text_test, predictions_NB)
    class_label = [0, 1]
    df_cm = pd.DataFrame(cm, index=class_label, columns=class_label)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap="crest")
    plt.title("Confusion Matrix - Naive Bayes")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig('figures/nb_' +
                str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + '.png')

    # lime explainer
    news_list = news_test.tolist()
    explainer = LimeTextExplainer(class_names=['True', 'False'], bow=True)
    explanation = explainer.explain_instance(news_list[3],
                                             pipeline_nb.predict_proba,
                                             num_features=20
                                             )
    explanation.save_to_file('html/nb_lime_explanation.html')
