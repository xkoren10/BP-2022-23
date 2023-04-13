# Python program to generate WordCloud
import sys

# importing all necessary modules
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


def make_cloud(args):

    df = open(args, "r", encoding="utf-8")

    stopwords = STOPWORDS
    comment_words = ''
    with open('../datasets/sk/stopwords.txt') as file:
        lines = [line.rstrip() for line in file]


# iterate through the csv file
    for val in df:

    # typecaste each val to string
        val = str(val)

    # split the value
        tokens = val.split()

    # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

        comment_words += " ".join(tokens) + " "

    comment_words.replace('\"','')



    wordcloud = WordCloud(width=800, height=800,
                            background_color='white',
                            stopwords=lines,
                            min_font_size=10).generate(comment_words)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()

    wc = WordCloud().generate(df.groupby('label')['title'].sum()[0])
    plt.figure(figsize=(15, 15))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")

