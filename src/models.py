# Python program to train NB, SVM, DT and SNN models
import pandas as pd
import NB_model
import SV_model
import DT_model
import snn
import saved_model
from sklearn.model_selection import train_test_split
import argparse





# Arguments parsing

parser = argparse.ArgumentParser(
    description='Python program to train NB, SVM, DT and SNN models')
model_group = parser.add_mutually_exclusive_group(required=True)
switch_group = parser.add_mutually_exclusive_group(required=True)

model_group.add_argument("-NB", "--NB", action="store_true",
                         dest="NBmodel", help='Naive Bayes')
model_group.add_argument("-DT", "--DT", action="store_true",
                         dest="DTmodel", help='Decision Tree')
model_group.add_argument("-SVM", "--SVM", action="store_true",
                         dest="SVMmodel", help='Support Vectors Machine')
model_group.add_argument("-SNN", "--SNN", action="store_true",
                         dest="SNNmodel", help='Sequential Neural Network')
switch_group.add_argument("-s", "--s", action="store_true",
                          dest="save", help='Save new model')
switch_group.add_argument("-u", "--u", action="store_true",
                          dest="used", help='Use pre-trained')

parser.add_argument("-url", "--url", action="store",
                    dest="url", default=None, help='Article url')

arguments = parser.parse_args()


# Dataset handling

# truenews1 = pd.read_csv('datasets/gossipcop_real.csv')
# fakenews1 = pd.read_csv('datasets/gossipcop_fake.csv')
# truenews2 = pd.read_csv('datasets/politifact_real.csv')
# fakenews2 = pd.read_csv('datasets/politifact_fake.csv')
# truenews1['True/Fake']='True'
# fakenews1['True/Fake']='Fake'
# truenews2['True/Fake']='True'
# fakenews2['True/Fake']='Fake'
# file = pd.concat([truenews1, fakenews1, truenews2, fakenews2])

# file['Article'] = file['title']
# label = file['True/Fake']


file = pd.read_csv("datasets/sk/dataset_sk_csv_utf.csv", sep=';')

if 'text' and 'title' in file.columns:
    file.dropna(subset=['text', 'title'], inplace=True)
    file['Article'] = file['text'] + ' ' + file['title']

label = file['label']

# Data splitting                                                                                        // static size
news_train, news_test, text_train, text_test = train_test_split(
    file['Article'], label, test_size=0.3)  # , random_state=1)


# Saving or using pre-trained
if arguments.used:
    saved_model.saved_model(news_test, arguments)
    exit(0)

# Models
if arguments.NBmodel:
    NB_model.nb_model(news_train, news_test, text_train,
                      text_test, arguments.save)

elif arguments.SVMmodel:
    SV_model.sv_model(news_train, news_test, text_train,
                      text_test,  arguments.save)

elif arguments.DTmodel:
    DT_model.dt_model(news_train, news_test, text_train,
                      text_test,  arguments.save)

elif arguments.SNNmodel:
    snn.snn_model(news_train, news_test, text_train,
                  text_test,  arguments.save)
