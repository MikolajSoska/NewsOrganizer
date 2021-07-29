import flask
from nltk.tokenize import sent_tokenize
from torchtext.data.utils import get_tokenizer

from utils.database import DatabaseConnector

app = flask.Flask(__name__)
connector = DatabaseConnector()
articles = connector.get_articles()
word_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')


@app.route('/news', methods=['GET'])
def get_articles():
    tags_count = connector.get_article_tags_count()
    return flask.render_template('news.html', articles=articles, tags_count=tags_count)


@app.route('/article/<article_index>', methods=['GET'])
def show_article(article_index: int):
    article = articles[int(article_index) - 1]
    content = []
    for sentence in sent_tokenize(article.content):
        for word in word_tokenizer(sentence):
            if len(content) in article.named_entities:
                word = f'<strong>{word} ({article.named_entities[len(content)]})</strong>'
            content.append(word)

    content = ' '.join(content)

    return flask.render_template('article.html', article=article, content=content)


app.run(host='127.0.0.1', port=5001)
