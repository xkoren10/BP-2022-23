# Python program to train NB, SVM, DT and SNN models
import sys
import pandas as pd
import NB_model, SV_model, DT_model, snn
from sklearn.model_selection import train_test_split


list_of_args = sys.argv

if len(list_of_args) != 2:
    exit(-1)
if list_of_args[1] is None or len(list_of_args)>2:
    exit (-1)

#https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/RBKVBM

# Dataset handling

#news = pd.read_csv('datasets/WELFake_Dataset.csv')
#fakenews = pd.read_csv('datasets/gossipcop_fake.csv')
#truenews['True/Fake']='True'
#fakenews['True/Fake']='Fake'
#news = pd.concat([truenews, fakenews])


file = pd.read_csv("datasets/WELFake_Dataset.csv")
file.dropna(subset = ['text', 'title'], inplace = True)
file['Article'] = file['text'] + ' ' + file['title']
label = file['label']
news_train, news_test, text_train, text_test = train_test_split(file['Article'], label, test_size=0.3)


# Models
if list_of_args[1] == '--NB':
    NB_model.nb_model(news_train, news_test, text_train, text_test)

elif list_of_args[1] == '--SV':
    SV_model.sv_model(news_train, news_test, text_train, text_test)

elif list_of_args[1] == '--DT':
    DT_model.dt_model(news_train, news_test, text_train, text_test)

elif list_of_args[1] == '--SNN':
    snn.snn_model(news_train, news_test, text_train, text_test)


