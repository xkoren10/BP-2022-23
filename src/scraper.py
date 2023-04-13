from newspaper import Article

# A new article from TOI


def parse_article(url):

    # For different language newspaper refer above table
    toi_article = Article(url)  # en for English

    # To download the article
    toi_article.download()

    # To parse the article
    toi_article.parse()

    full_text = toi_article.title + toi_article.text  # somtimes only title??

    full_text = full_text.replace('\n', '')

    full_text = [full_text]

    return full_text
