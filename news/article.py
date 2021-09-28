from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any


@dataclass(init=True, repr=True, eq=False)
class Country:
    name: str
    code: str
    language: str


@dataclass(init=True, repr=True, eq=False)
class NewsSite:
    name: str
    code: str
    country: Country


@dataclass(init=True, repr=True, eq=False)
class NamedEntity:
    full_name: str
    short_name: str
    position: int
    length: int
    words: str


@dataclass(init=True, repr=True, eq=False)
class NewsArticle:
    article_id: int
    title: str
    content: str  # TODO: change back to list with mechanism to restore division into paragraphs
    article_url: str
    article_date: datetime
    news_site: NewsSite
    image_url: str
    summaries: Dict[int, str] = field(default_factory=dict)
    named_entities: Dict[int, List[NamedEntity]] = field(default_factory=dict)


@dataclass(init=True, repr=True, eq=False)
class NewsModel:
    model_id: int
    fullname: str
    name_identifier: str
    class_name: str
    constructor_args: Dict[str, Any]
    dataset_args: Dict[str, Any]
    batch_size: int
    dataset_name: str
