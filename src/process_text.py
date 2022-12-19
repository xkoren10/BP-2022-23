import string
from nltk.corpus import stopwords

def process_text(s):

    nopunc = [char for char in s if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    #
    clean_string = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_string