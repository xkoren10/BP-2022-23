# Python program to train NB, SVM and DT models
import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import NB_model, SV_model, DT_model
import process_text as pt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


list_of_args = sys.argv

if len(list_of_args) != 2:
    exit(-1)
if list_of_args[1] is None or len(list_of_args)>2:
    exit (-1)



truenews = pd.read_csv('datasets/True.csv')[:200]
fakenews = pd.read_csv('datasets/Fake.csv')[:200]

# Dataset showcases
#print(truenews)
#print(fakenews)


truenews['True/Fake']='True'
fakenews['True/Fake']='Fake'

# Combine the 2 DataFrames into a single data frame
news = pd.concat([truenews, fakenews])
news["Article"] = news["title"] + news["text"]


# Tokenization - tokens are words that we actually want
news['Clean Text'] = news['Article'].apply(pt.process_text)

bow_transformer = CountVectorizer(analyzer=pt.process_text).fit(news['Clean Text'])
# python package, popisat vektorizaciu, velkost vektorov, vizualizacia (2D, PCA)

# Bag-of-Words (bow)
news_bow = bow_transformer.transform(news['Clean Text'])


print('Shape of Sparse Matrix: ', news_bow.shape)
print('Amount of Non-Zero occurences: ', news_bow.nnz)


sparsity = (100.0 * news_bow.nnz / (news_bow.shape[0] * news_bow.shape[1]))
print('Sparisty: {}'.format(round(sparsity)))



# TF-IDF
tfidf_transformer = TfidfTransformer().fit(news_bow)
news_tfidf = tfidf_transformer.transform(news_bow)

# TODO PCA
#pca = PCA(n_components=2).fit(news_tfidf)
#data2D = pca.transform(news_tfidf)
#plt.scatter(data2D[:,0], data2D[:,1],c=news['Clean Text'])
#plt.show()

# Models
if list_of_args[1] == '--NB':
    NB_model.nb_model(news_tfidf,news)

elif list_of_args[1] == '--SV':
    SV_model.sv_model(news_tfidf, news)

elif list_of_args[1] == '--DT':
    DT_model.dt_model(news_tfidf, news)

exit(0)
