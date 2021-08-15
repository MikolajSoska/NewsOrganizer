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
    `image_url`    varchar(255) NOT NULL,
    `summary`      text
);

CREATE TABLE `datasets`
(
    `id`   int PRIMARY KEY AUTO_INCREMENT,
    `name` varchar(255) UNIQUE NOT NULL
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
    `category_id` int NOT NULL
);

CREATE TABLE `article_tag_map`
(
    `id`              int PRIMARY KEY AUTO_INCREMENT,
    `article_id`      int          NOT NULL,
    `tag_category_id` int          NOT NULL,
    `position`        int          NOT NULL,
    `length`          int          NOT NULL,
    `words`           varchar(255) NOT NULL
);

ALTER TABLE `countries`
    ADD FOREIGN KEY (`language_id`) REFERENCES `languages` (`id`);

ALTER TABLE `news_sites`
    ADD FOREIGN KEY (`country_id`) REFERENCES `countries` (`id`);

ALTER TABLE `news_articles`
    ADD FOREIGN KEY (`site_id`) REFERENCES `news_sites` (`id`);

ALTER TABLE `tag_categories`
    ADD FOREIGN KEY (`dataset_id`) REFERENCES `datasets` (`id`);

ALTER TABLE `tags`
    ADD FOREIGN KEY (`category_id`) REFERENCES `tag_categories` (`id`);

ALTER TABLE `article_tag_map`
    ADD FOREIGN KEY (`article_id`) REFERENCES `news_articles` (`id`);

ALTER TABLE `article_tag_map`
    ADD FOREIGN KEY (`tag_category_id`) REFERENCES `tag_categories` (`id`);

CREATE UNIQUE INDEX `tag_categories_index_0` ON `tag_categories` (`category_name`, `dataset_id`);

CREATE UNIQUE INDEX `tags_index_1` ON `tags` (`tag`, `category_id`);

CREATE UNIQUE INDEX `tags_index_2` ON `tags` (`tag_label`, `category_id`);

CREATE UNIQUE INDEX `article_tag_map_index_3` ON `article_tag_map` (`article_id`, `position`);
