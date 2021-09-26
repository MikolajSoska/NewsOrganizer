INSERT INTO languages
VALUES (0, 'English', 'en');

INSERT INTO countries
VALUES (0, 'United States', 'us', 1);

INSERT INTO news_sites
VALUES (0, 'CNN', 'cnn', 1),
       (0, 'Politico', 'politico', 1),
       (0, 'Newsweek', 'newsweek', 1),
       (0, 'National Geographic', 'national-geographic', 1);

INSERT INTO tasks
VALUES (0, 'Named Entity Recognition'),
       (0, 'Abstractive Summarization');

INSERT INTO datasets
VALUES (0, 'conll2003', 'CoNLL-2003', 1),
       (0, 'gmb', 'GMB', 1),
       (0, 'cnn_dailymail', 'CNN/Daily Mail', 1),
       (0, 'xsum', 'XSum', 1);

INSERT INTO models
VALUES (0, 'BiLSTM-CNN', 'bilstm_cnn', 1, 1),
       (0, 'BiLSTM-CNN', 'bilstm_cnn-gmb-new', 2, 1),
       (0, 'BiLSTM-CRF', 'bilstm_crf', 1, 1),
       (0, 'BiLSTM-CRF', 'bilstm_crf-gmb9', 2, 1),
       (0, 'ID-CNN', 'id_cnn', 1, 1),
       (0, 'ID-CNN', 'id_cnn-gmb-final', 2, 1),
       (0, 'Pointer-generator', 'pointer_generator', 3, 2),
       (0, 'Pointer-generator', 'pointer_generator-xsum', 4, 2),
       (0, 'RL+ML', 'reinforcement_learning', 3, 2),
       (0, 'RL+ML', 'reinforcement_learning-xsum-mixed', 4, 2),
       (0, 'Transformer', 'transformer', 3, 2),
       (0, 'Transformer', 'transformer-xsum', 4, 2);


# CoNLL2003 dataset
INSERT INTO tag_categories
VALUES (0, 'Persons', 1),
       (0, 'Organizations', 1),
       (0, 'Locations', 1),
       (0, 'Miscellaneous', 1);

INSERT INTO tags
VALUES (0, 'B-PER', 1, 1),
       (0, 'I-PER', 2, 1),
       (0, 'B-ORG', 3, 2),
       (0, 'I-ORG', 4, 2),
       (0, 'B-LOC', 5, 3),
       (0, 'I-LOC', 6, 3),
       (0, 'B-MISC', 7, 4),
       (0, 'I-MISC', 8, 4);


# GMB dataset
INSERT INTO tag_categories
VALUES (0, 'Geographical Entities', 2),
       (0, 'Organizations', 2),
       (0, 'Persons', 2),
       (0, 'Geopolitical Entities', 2),
       (0, 'Time indicator', 2),
       (0, 'Artifact', 2),
       (0, 'Event', 2),
       (0, 'Natural Phenomenon', 2);


INSERT INTO tags
VALUES (0, 'B-GEO', 1, 5),
       (0, 'I-GEO', 2, 5),
       (0, 'B-ORG', 3, 6),
       (0, 'I-ORG', 4, 6),
       (0, 'B-PER', 5, 7),
       (0, 'I-PER', 6, 7),
       (0, 'B-GPE', 7, 8),
       (0, 'I-GPE', 8, 8),
       (0, 'B-TIM', 9, 9),
       (0, 'I-TIM', 10, 9),
       (0, 'B-ART', 11, 10),
       (0, 'I-ART', 12, 10),
       (0, 'B-EVE', 13, 11),
       (0, 'I-EVE', 14, 11),
       (0, 'B-NAT', 15, 12),
       (0, 'I-NAT', 16, 12);