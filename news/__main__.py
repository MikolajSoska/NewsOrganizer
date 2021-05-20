import argparse

import tqdm

from database import DatabaseConnector
from neural.predict import NewsPredictor
from news.getter import NewsGetter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Get current news articles and add to database')
    parser.add_argument('--api-key', dest='api_key', type=str, help='API key for NewsAPI', required=True)

    return parser.parse_args()


def main(news_api_key: str):
    connector = DatabaseConnector()
    news_getter = NewsGetter(news_api_key)
    predictor = NewsPredictor(use_cuda=True)

    for country in connector.get_countries():
        print(f'Adding news articles for {country.name}...')
        articles = news_getter.get_articles(country)
        for article in tqdm.tqdm(articles, desc='Processing articles and adding to database'):
            article = predictor.process_article(article)
            connector.add_new_article(article)
        print(f'Added {len(articles)} new articles.')


if __name__ == '__main__':
    args = parse_args()
    main(args.api_key)
