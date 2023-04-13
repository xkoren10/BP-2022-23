from scikeras.wrappers import KerasClassifier
import keras
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import process_text as pt
import string
from datetime import datetime
import dill as pickle
from lime.lime_text import LimeTextExplainer




def snn_model(news_train, news_test, text_train, text_test, saved):

    def make_model():

        model = keras.models.Sequential()  # Model initialization.
        model.add(keras.layers.Dense(128, input_shape=(
            None, 9973), activation='relu'))  # shape input
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(2, activation='softmax'))  # FAKE/REAL
        model.compile(
            optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    # es = EarlyStopping(monitor='loss', mode='min', verbose=1)
    clf = KerasClassifier(build_fn=make_model, verbose=1,
                          epochs=10, batch_size=10)  # , callbacks=[es])

    pipeline_snn = Pipeline([
        # strings to token integer counts
        ('bow', CountVectorizer(analyzer=pt.process_text)),
        # integer counts to weighted TF-IDF scores
        ('tfidf', TfidfTransformer()),
        ('classifier', clf),  # train on Keras Sequential classifier
    ])

# staticka dlzka vektorov na vstupe (vybrat prvych n vzoriek)

    pipeline_snn.fit(news_train, text_train)
    predictions_snn = pipeline_snn.predict(news_test)  # Model predictions

    if saved == True:
        filename = 'model_snn_sk.pk'
        # save model when training on new dataset
        with open('saved_models/'+filename, 'wb') as file:
            pickle.dump(pipeline_snn, file)

    print('SNN - test')
    print(classification_report(predictions_snn, text_test))

    cm = confusion_matrix(text_test, predictions_snn)
    class_label = [0, 1]
    df_cm = pd.DataFrame(cm, index=class_label, columns=class_label)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap="crest")
    plt.title("Confusion Matrix - Sequential NN")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig('figures/snn_' +
                str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + '.png')

    # lime explainer
    news_list = news_test.tolist()
    explainer = LimeTextExplainer(class_names=['True', 'False'], bow=True)
    explanation = explainer.explain_instance(news_list[3],
                                             pipeline_snn.predict_proba,
                                             num_features=20
                                             )
    explanation.save_to_file('html/snn_lime_explanation.html')
