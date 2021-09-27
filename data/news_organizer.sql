CREATE TABLE `languages`
(
    `id`   int PRIMARY KEY AUTO_INCREMENT,
    `name` varchar(255) UNIQUE NOT NULL,
    `code` varchar(255) UNIQUE NOT NULL
);

CREATE TABLE `countries`
(
    `id`          int PRIMARY KEY AUTO_INCREMENT,
    `name`        varchar(255) UNIQUE NOT NULL,
    `code`        varchar(255) UNIQUE NOT NULL,
    `language_id` int                 NOT NULL
);

CREATE TABLE `news_sites`
(
    `id`         int PRIMARY KEY AUTO_INCREMENT,
    `name`       varchar(255) UNIQUE NOT NULL,
    `code`       varchar(255) UNIQUE NOT NULL,
    `country_id` int                 NOT NULL
);

CREATE TABLE `news_articles`
(
    `id`           int PRIMARY KEY AUTO_INCREMENT,
    `title`        varchar(255) NOT NULL,
    `content`      text         NOT NULL,
    `article_url`  varchar(255) NOT NULL,
    `article_date` datetime     NOT NULL,
    `site_id`      int          NOT NULL,
    `image_url` varchar(255) NOT NULL
);

CREATE TABLE `summaries`
(
    `id`         int PRIMARY KEY AUTO_INCREMENT,
    `content`    text NOT NULL,
    `article_id` int  NOT NULL,
    `model_id`   int  NOT NULL
);

CREATE TABLE `tag_categories`
(
    `id`            int PRIMARY KEY AUTO_INCREMENT,
    `category_name` varchar(255) UNIQUE NOT NULL,
    `dataset_id`    int                 NOT NULL
);

CREATE TABLE `tags`
(
    `id`          int PRIMARY KEY AUTO_INCREMENT,
    `tag`         varchar(255) NOT NULL,
    `tag_label`   int          NOT NULL,
    `category_id` int          NOT NULL
);

CREATE TABLE `article_tag_map`
(
    `id`              int PRIMARY KEY AUTO_INCREMENT,
    `tag_category_id` int          NOT NULL,
    `model_id`        int          NOT NULL,
    `article_id`      int          NOT NULL,
    `position`        int          NOT NULL,
    `length`          int          NOT NULL,
    `words`           varchar(255) NOT NULL
);

CREATE TABLE `tasks`
(
    `id`   int PRIMARY KEY AUTO_INCREMENT,
    `name` varchar(255) UNIQUE NOT NULL
);

CREATE TABLE `datasets`
(
    `id`          int PRIMARY KEY AUTO_INCREMENT,
    `id_name`     varchar(255) UNIQUE NOT NULL,
    `full_name`   varchar(255) UNIQUE NOT NULL,
    `language_id` int                 NOT NULL,
    `task_id`     int                 NOT NULL
);

CREATE TABLE `models`
(
    `id`               int PRIMARY KEY AUTO_INCREMENT,
    `model_name`       varchar(255) NOT NULL,
    `model_identifier` varchar(255) NOT NULL,
    `class_name`       varchar(255) NOT NULL
);

CREATE TABLE `news_models`
(
    `id`               int PRIMARY KEY AUTO_INCREMENT,
    `model_id`         int  NOT NULL,
    `dataset_id`       int  NOT NULL,
    `constructor_args` blob NOT NULL,
    `dataset_args`     blob NOT NULL,
    `batch_size`       int  NOT NULL
);

ALTER TABLE `countries`
    ADD FOREIGN KEY (`language_id`) REFERENCES `languages` (`id`);

ALTER TABLE `news_sites`
    ADD FOREIGN KEY (`country_id`) REFERENCES `countries` (`id`);

ALTER TABLE `news_articles`
    ADD FOREIGN KEY (`site_id`) REFERENCES `news_sites` (`id`);

ALTER TABLE `summaries`
    ADD FOREIGN KEY (`article_id`) REFERENCES `news_articles` (`id`);

ALTER TABLE `summaries`
    ADD FOREIGN KEY (`model_id`) REFERENCES `news_models` (`id`);

ALTER TABLE `tag_categories`
    ADD FOREIGN KEY (`dataset_id`) REFERENCES `datasets` (`id`);

ALTER TABLE `tags`
    ADD FOREIGN KEY (`category_id`) REFERENCES `tag_categories` (`id`);

ALTER TABLE `article_tag_map`
    ADD FOREIGN KEY (`article_id`) REFERENCES `news_articles` (`id`);

ALTER TABLE `article_tag_map`
    ADD FOREIGN KEY (`tag_category_id`) REFERENCES `tag_categories` (`id`);

ALTER TABLE `article_tag_map`
    ADD FOREIGN KEY (`model_id`) REFERENCES `news_models` (`id`);

ALTER TABLE `datasets`
    ADD FOREIGN KEY (`language_id`) REFERENCES `languages` (`id`);

ALTER TABLE `datasets`
    ADD FOREIGN KEY (`task_id`) REFERENCES `tasks` (`id`);

ALTER TABLE `news_models`
    ADD FOREIGN KEY (`dataset_id`) REFERENCES `datasets` (`id`);

ALTER TABLE `news_models`
    ADD FOREIGN KEY (`model_id`) REFERENCES `models` (`id`);

CREATE UNIQUE INDEX `summaries_index_0` ON `summaries` (`article_id`, `model_id`);

CREATE UNIQUE INDEX `tag_categories_index_1` ON `tag_categories` (`category_name`, `dataset_id`);

CREATE UNIQUE INDEX `tags_index_2` ON `tags` (`tag`, `category_id`);

CREATE UNIQUE INDEX `tags_index_3` ON `tags` (`tag_label`, `category_id`);

CREATE UNIQUE INDEX `article_tag_map_index_4` ON `article_tag_map` (`article_id`, `model_id`, `position`);

CREATE UNIQUE INDEX `datasets_index_5` ON `datasets` (`id_name`, `language_id`);

CREATE UNIQUE INDEX `datasets_index_6` ON `datasets` (`full_name`, `language_id`);

CREATE UNIQUE INDEX `datasets_index_7` ON `datasets` (`id_name`, `task_id`);

CREATE UNIQUE INDEX `models_index_8` ON `models` (`model_name`, `model_identifier`, `class_name`);

CREATE UNIQUE INDEX `news_models_index_9` ON `news_models` (`model_id`, `dataset_id`);
