import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def process_text(text,lang='slovak'):
    text = ''.join(
        [c for c in text if c not in string.punctuation and c not in string.digits])
    tokens = word_tokenize(text)
    lemmatiser = WordNetLemmatizer()
    lemmatized = [lemmatiser.lemmatize(word) for word in tokens]

    if lang == 'slovak':
        with open('datasets/sk/stopwords.txt') as file:
            sw = [line.rstrip() for line in file]
    else :
        sw = stopwords.words(lang)

    stopped = [word for word in lemmatized if word.lower() not in sw]
    return stopped
