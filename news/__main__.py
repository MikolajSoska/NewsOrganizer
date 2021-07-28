import argparse
import sys

import tqdm
from torchtext.data.utils import get_tokenizer

from database import DatabaseConnector
from neural.common.utils import tokenize_text_content
from neural.predict import NewsPredictor
from news.getter import NewsGetter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Get current news articles and add to database')
    parser.add_argument('--api-key', dest='api_key', type=str, help='API key for NewsAPI', required=True)

    return parser.parse_args()


def main(news_api_key: str) -> None:
    connector = DatabaseConnector()
    news_getter = NewsGetter(news_api_key)
    predictor = NewsPredictor(use_cuda=True)
    word_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

    for country in connector.get_countries():
        print(f'Adding news articles for {country.name}...')
        articles = news_getter.get_articles(country)
        for article in tqdm.tqdm(articles, desc='Processing articles and adding to database', file=sys.stdout):
            article = predictor.process_article(article)
            tokens = tokenize_text_content(article.content, word_tokenizer=word_tokenizer)
            connector.add_new_article(article, tokens)
        print(f'Added {len(articles)} new articles.')


if __name__ == '__main__':
    args = parse_args()
    main(args.api_key)
