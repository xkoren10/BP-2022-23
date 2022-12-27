# Python program to train NB, SVM and DT models
import sys
import pandas as pd
import NB_model, SV_model, DT_model
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA



list_of_args = sys.argv

if len(list_of_args) != 2:
    exit(-1)
if list_of_args[1] is None or len(list_of_args)>2:
    exit (-1)


# Dataset handling
news = pd.read_csv('datasets/WELFake_Dataset.csv',index_col=0)
#fakenews = pd.read_csv('datasets/gossipcop_fake.csv')

news = news.dropna()
news["Article"] = news["title"] + news["text"]
news_train, news_test, text_train, text_test = train_test_split(news['Article'], news['label'], test_size=0.3)


# Models
if list_of_args[1] == '--NB':
    NB_model.nb_model(news_train, news_test, text_train, text_test)

elif list_of_args[1] == '--SV':
    SV_model.sv_model(news_train, news_test, text_train, text_test)

elif list_of_args[1] == '--DT':
    DT_model.dt_model(news_train, news_test, text_train, text_test)


