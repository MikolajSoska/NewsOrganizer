import json

import requests


class NewsGetter:
    __API_URL = 'https://newsapi.org/v2'

    def __init__(self, api_key: str):
        self.__session = requests.Session()
        self.__session.headers.update({'Authorization': f'{api_key}'})

    def get_top_articles(self, country: str = 'us') -> None:
        url = f'{NewsGetter.__API_URL}/top-headlines'
        parameters = {
            'country': country,
            'pageSize': 100
        }

        response = self.__session.get(url, params=parameters)
        if response.status_code == requests.codes.ok:
            response = response.json()
            print(json.dumps(response, indent=5))
        else:
            print(f'HTTP {response.status_code} code when getting news articles. Error message: {response.text}.')
