import flask
from torchtext.data.utils import get_tokenizer

from utils.database import DatabaseConnector
from utils.general import tokenize_text_content

app = flask.Flask(__name__)
connector = DatabaseConnector()
word_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
ner_models = connector.get_models_by_task('Named Entity Recognition', full_dataset_name=True)
summarization_models = connector.get_models_by_task('Abstractive Summarization', full_dataset_name=True)


@app.route('/', methods=['GET'])
def get_home():
    return flask.redirect(flask.url_for('get_articles'))


@app.route('/news', methods=['GET'])
def get_articles():
    articles = connector.get_articles()
    summaries = connector.get_articles_summaries(summarization_models[0].model_id)
    named_entities = connector.get_articles_top_named_entities(ner_models[0].model_id)
    tags_count = connector.get_articles_tags_count(ner_models[0].model_id)
    return flask.render_template('news.html', articles=articles, summaries=summaries, named_entities=named_entities,
                                 tags_count=tags_count, ner_models=ner_models,
                                 summarization_models=summarization_models)


@app.route('/news-filter', methods=['GET'])
def get_articles_filter():
    model_id = flask.request.args.get('model_id', type=int)
    tags = flask.request.args.get('tags', type=str)
    if len(tags) == 0:
        model_id = None  # Disable filtering when no tag is provided
    tags = tuple(tags.split('\t'))

    articles = connector.get_articles(model_id, tags)
    return flask.jsonify([article.article_id for article in articles])


@app.route('/article-entities/<int:article_id>/<int:model_id>', methods=['GET'])
def get_articles_entities(article_id: int, model_id: int):
    article = connector.get_single_article(article_id)
    named_entities = connector.get_article_named_entities(model_id, article_id)
    tokens = tokenize_text_content(article.content, word_tokenizer=word_tokenizer)

    for entity in named_entities:
        start_token = tokens[entity.position]
        tokens[entity.position] = f'<div class="btn btn-sm article-tag tag-{entity.short_name.lower()}">{start_token}'
        last_position = entity.position + entity.length - 1
        tokens[last_position] = f'{tokens[last_position]} <span class="badge">{entity.short_name}</span></div>'

    content = ' '.join(tokens)

    return flask.jsonify(content)


@app.route('/summaries/<int:model_id>', methods=['GET'])
def get_summaries(model_id: int):
    summaries = connector.get_articles_summaries(model_id)
    return flask.jsonify(summaries)


@app.route('/tags-count/<int:model_id>', methods=['GET'])
def get_tags_count(model_id: int):
    tags_count = connector.get_articles_tags_count(model_id)
    return flask.jsonify(tags_count)


@app.route('/named-entities/<int:model_id>', methods=['GET'])
def get_named_entities(model_id: int):
    named_entities = connector.get_articles_top_named_entities(model_id)
    return flask.jsonify(named_entities)


app.run(host='127.0.0.1', port=5001)
