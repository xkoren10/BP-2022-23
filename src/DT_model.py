from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
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
from sklearn.tree import plot_tree
from lime.lime_text import LimeTextExplainer


def dt_model(news_train, news_test, text_train, text_test, saved):

    pipeline_dt = Pipeline([
        # strings to token integer counts
        ('bow', CountVectorizer(analyzer=pt.process_text)),
        # integer counts to weighted TF-IDF scores
        ('tfidf', TfidfTransformer()),
        # train on TF-IDF vectors w/ Decision Tree classifier
        ('classifier', DecisionTreeClassifier(min_samples_leaf=10)),
    ])

    pipeline_dt.fit(news_train, text_train)
    predictions_DT = pipeline_dt.predict(news_test)

    # tree plotting
    plt.figure()
    plot_tree(pipeline_dt['classifier'], class_names=['True', 'False'])
    plt.savefig('figures/tree.png', format='png', bbox_inches="tight")

    print('DT - test')
    print(classification_report(predictions_DT, text_test))

    if saved == True:
        filename = 'saved_models/model_dt.pk'
        with open(filename, 'wb') as file:       # save model when training on new dataset
            pickle.dump(pipeline_dt, file)

    cm = confusion_matrix(text_test, predictions_DT)
    class_label = [0, 1]

    plt.figure()
    df_cm = pd.DataFrame(cm, index=class_label, columns=class_label)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap="crest")
    plt.title("Confusion Matrix - Decision Tree")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig('figures/dt_' +
                str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + '.png')

    # lime explainer

    news_list = news_test.tolist()
    explainer = LimeTextExplainer(class_names=['True', 'False'], bow=True)
    explanation = explainer.explain_instance(news_list[3],
                                             pipeline_dt.predict_proba,
                                             num_features=20
                                             )
    explanation.save_to_file('html/dt_lime_explanation.html')
