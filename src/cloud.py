# Python program to generate WordCloud
import sys

# importing all necessary modules
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


def make_cloud(args):

    df = open(args, "r", encoding="utf-8")

    comment_words = ''
    stopwords = set(STOPWORDS)
    stopwords.add("S")


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

    wordcloud = WordCloud(width=800, height=800,
                            background_color='white',
                            stopwords=stopwords,
                            min_font_size=10).generate(comment_words)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()