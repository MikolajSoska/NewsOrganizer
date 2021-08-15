INSERT INTO languages
VALUES (0, 'English', 'en');

INSERT INTO countries
VALUES (0, 'United States', 'us', 1);

INSERT INTO news_sites
VALUES (0, 'CNN', 'cnn', 1),
       (0, 'Politico', 'politico', 1),
       (0, 'Newsweek', 'newsweek', 1),
       (0, 'National Geographic', 'national-geographic', 1);

INSERT INTO datasets
VALUES (0, 'conll2003');

INSERT INTO tag_categories
VALUES (0, 'Persons'),
       (0, 'Organizations'),
       (0, 'Locations'),
       (0, 'Miscellaneous');

INSERT INTO tags
VALUES (0, 'B-PER', 1, 1, 1),
       (0, 'I-PER', 2, 1, 1),
       (0, 'B-ORG', 3, 2, 1),
       (0, 'I-ORG', 4, 2, 1),
       (0, 'B-LOC', 5, 3, 1),
       (0, 'I-LOC', 6, 3, 1),
       (0, 'B-MISC', 7, 4, 1),
       (0, 'I-MISC', 8, 4, 1);

