import flask
from torchtext.data.utils import get_tokenizer

from utils.database import DatabaseConnector
from utils.general import tokenize_text_content

app = flask.Flask(__name__)
connector = DatabaseConnector()
articles = connector.get_articles()
articles_map = {article.article_id: article for article in articles}
word_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
ner_models = connector.get_models_by_task('Named Entity Recognition', full_dataset_name=True)
summarization_models = connector.get_models_by_task('Abstractive Summarization', full_dataset_name=True)
tags_count = connector.get_article_tags_count()


@app.route('/', methods=['GET'])
def get_home():
    return flask.redirect(flask.url_for('get_articles'))


@app.route('/news', methods=['GET'])
def get_articles():
    summaries = connector.get_articles_summaries(summarization_models[0].model_id)
    named_entities = connector.get_articles_top_named_entities(ner_models[0].model_id)
    return flask.render_template('news.html', articles=articles, summaries=summaries, named_entities=named_entities,
                                 tags_count=tags_count, ner_models=ner_models,
                                 summarization_models=summarization_models)


@app.route('/article/<int:article_id>', methods=['GET'])
def show_article(article_id: int):
    article = articles_map[article_id]
    tokens = tokenize_text_content(article.content, word_tokenizer=word_tokenizer)

    for tag, position, length, _ in article.named_entities:
        start_token = tokens[position]
        tokens[position] = f'<strong>{start_token}'
        end_token = tokens[position + length - 1]
        tokens[position + length - 1] = f'{end_token} ({tag})</strong>'

    content = ' '.join(tokens)

    return flask.render_template('article.html', article=article, content=content)


@app.route('/summaries/<int:model_id>', methods=['GET'])
def get_summaries(model_id: int):
    summaries = connector.get_articles_summaries(model_id)
    return flask.jsonify(summaries)


@app.route('/named-entities/<int:model_id>', methods=['GET'])
def get_named_entities(model_id: int):
    named_entities = connector.get_articles_top_named_entities(model_id)
    return flask.jsonify(named_entities)


app.run(host='127.0.0.1', port=5001)
