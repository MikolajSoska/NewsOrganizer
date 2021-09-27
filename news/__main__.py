import argparse
import sys

import tqdm

from neural.predict import NewsPredictor
from news.getter import NewsGetter
from utils.database import DatabaseConnector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Get current news articles and add to database')
    parser.add_argument('--api-key', type=str, help='API key for NewsAPI', required=True)
    parser.add_argument('--models-path', type=str, help='Path to saved models', default='../data/saved/predictor')
    parser.add_argument('--vocab-path', type=str, help='Path to saved vocabs', default='../data/saved/vocabs')
    parser.add_argument('--summarization-vocab-size', type=int, help='Summarization vocab size', default=50000)
    parser.add_argument('--use-gpu', action='store_true', help='Run predictor with CUDA')

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    connector = DatabaseConnector()
    news_getter = NewsGetter(args.api_key)
    predictor = NewsPredictor(
        summarization_vocab_size=args.summarization_vocab_size,
        use_cuda=args.use_gpu,
        path_to_models=args.models_path,
        path_to_vocabs=args.vocab_path
    )

    for country in connector.get_countries():
        print(f'Adding news articles for {country.name}...')
        articles = news_getter.get_articles(country)
        predictor.process_articles(articles, model_type='NER')
        predictor.process_articles(articles, model_type='summarization')

        for article in tqdm.tqdm(articles, desc='Adding articles to database', file=sys.stdout):
            article_id = connector.add_new_article(article)
            connector.add_article_summaries(article_id, article.summaries)
            connector.add_article_names_entities(article_id, article.named_entities)
        print(f'Added {len(articles)} new articles.')


if __name__ == '__main__':
    main()
