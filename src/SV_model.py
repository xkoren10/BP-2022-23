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
import dill as pickle
from datetime import datetime
from lime.lime_text import LimeTextExplainer

# Tensorflow warning silencer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def sv_model(news_train, news_test, text_train, text_test, saved):

    pipeline_sv = Pipeline([
        # strings to token integer counts
        ('bow', CountVectorizer(analyzer=pt.process_text)),
        # integer counts to weighted TF-IDF scores
        ('tfidf', TfidfTransformer()),
        # train w/ Support Vectors classifier
        ('classifier', SVC(C=1.0, kernel='linear', gamma='auto', probability=True)),
    ])

    pipeline_sv.fit(news_train, text_train)
    predictions_SVM = pipeline_sv.predict(news_test)

    if saved == True:
        filename = 'saved_models/model_svm.pk'
        with open(filename, 'wb') as file:       # save model when training on new dataset
            pickle.dump(pipeline_sv, file)

    print('SVM - test')
    print(classification_report(predictions_SVM, text_test))

    cm = confusion_matrix(text_test, predictions_SVM)
    class_label = [0, 1]
    df_cm = pd.DataFrame(cm, index=class_label, columns=class_label)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap="crest")
    plt.title("Confusion Matrix - Support vectors")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig('figures/svm' +
                str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + '.png')

    # lime explainer
    news_list = news_test.tolist()
    explainer = LimeTextExplainer(class_names=['True', 'False'], bow=True)
    explanation = explainer.explain_instance(news_list[3],
                                             pipeline_sv.predict_proba,
                                             num_features=20
                                             )
    explanation.save_to_file('html/svm_lime_explanation.html')
