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

CREATE TABLE `tags`
(
    `id`        int PRIMARY KEY AUTO_INCREMENT,
    `tag_short` varchar(255) UNIQUE NOT NULL,
    `fullname`  varchar(255) UNIQUE NOT NULL
);

CREATE TABLE `article_tag_map`
(
    `id`         int PRIMARY KEY AUTO_INCREMENT,
    `article_id` int NOT NULL,
    `tag_id`     int NOT NULL,
    `position`   int NOT NULL
);

ALTER TABLE `countries`
    ADD FOREIGN KEY (`language_id`) REFERENCES `languages` (`id`);

ALTER TABLE `news_sites`
    ADD FOREIGN KEY (`country_id`) REFERENCES `countries` (`id`);

ALTER TABLE `news_articles`
    ADD FOREIGN KEY (`site_id`) REFERENCES `news_sites` (`id`);

ALTER TABLE `article_tag_map`
    ADD FOREIGN KEY (`article_id`) REFERENCES `news_articles` (`id`);

ALTER TABLE `article_tag_map`
    ADD FOREIGN KEY (`tag_id`) REFERENCES `tags` (`id`);

CREATE UNIQUE INDEX `article_tag_map_index_0` ON `article_tag_map` (`article_id`, `tag_id`, `position`);
