from scikeras.wrappers import KerasClassifier
import tensorflow.python.keras.backend as K
sess = K.get_session()
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

def snn_model (news_train, news_test, text_train, text_test):

    def make_model():

        model = keras.models.Sequential()  # Model initialization.
        model.add(keras.layers.Flatten())    # Flatten model. -> 1-D array like.
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(2, activation='softmax'))  # FAKE/REAL
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    clf = KerasClassifier(build_fn=make_model, verbose=1, epochs=100, batch_size=16, callbacks=[es])


    pipeline4 = Pipeline([
        ('bow', CountVectorizer(analyzer=pt.process_text)),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', clf),  # train on Keras Sequential classifier
    ])


    pipeline4.fit(news_train.to_numpy(), text_train.to_numpy())
    predictions_snn= pipeline4.predict(news_test)  # Model predictions

    print('SNN - test')
    print(classification_report(predictions_snn, text_test))

    cm = confusion_matrix(text_test, predictions_snn)
    class_label = [0, 1]
    df_cm = pd.DataFrame(cm, index=class_label, columns=class_label)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap="crest")
    plt.title("Confusion Matrix - Decision Tree")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()