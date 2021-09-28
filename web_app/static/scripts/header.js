$(document).ready(function () {
    $("#summarization-select").on("change", function () {
        prepareForNewSummaries();
        const summarizationSelect = document.getElementById("summarization-select");
        const modelID = summarizationSelect.value.replace("summarization-model-", "");
        const baseURL = window.location.origin;

        $.getJSON(baseURL + "/summaries/" + modelID, function (summaries) {
            for (const [articleID, summary] of Object.entries(summaries)) {
                const article = document.getElementById("article-" + articleID);
                const summaryNode = article.getElementsByClassName("summary")[0];
                summaryNode.innerHTML = String(summary);
            }
        })
    });
    $("#ner-select").on("change", function () {
        prepareForNewNamedEntities()
        const nerSelect = document.getElementById("ner-select");
        const modelID = nerSelect.value.replace("ner-model-", "");
        const baseURL = window.location.origin;

        $.getJSON(baseURL + "/named-entities/" + modelID, function (namedEntities) {
            for (const [articleID, named_entities] of Object.entries(namedEntities)) {
                let tagHTML = "";
                for (const [words, entity] of named_entities) {
                    tagHTML += "<div class=\"article-card-entity tag-" + entity.toLowerCase() + " btn\">";
                    tagHTML += words + " (" + entity + ")</div>";
                }
                const article = document.getElementById("article-" + articleID);
                const nerNode = article.getElementsByClassName("named-entities")[0];
                nerNode.innerHTML = String(tagHTML);
            }
        })
    });
});

function prepareForNewSummaries() {
    const summaryPlaceholder =
        "<div class=\"placeholder-glow\">\n" +
        " <span class=\"placeholder col-7\"></span>" +
        " <span class=\"placeholder col-4\"></span>" +
        " <span class=\"placeholder col-4\"></span>" +
        " <span class=\"placeholder col-2\"></span>" +
        " <span class=\"placeholder col-3\"></span>" +
        " <span class=\"placeholder col-8\"></span>" +
        " <span class=\"placeholder col-2\"></span>";

    const articles = document.getElementById("articles");
    for (const article of Array.from(articles.children)) {
        const summary = article.getElementsByClassName("summary")[0];
        summary.innerHTML = summaryPlaceholder;
    }
}

function prepareForNewNamedEntities() {
    const nerPlaceholder = "<button class=\"btn\" disabled>\n" +
        "<span class=\"spinner-grow spinner-grow-sm\" role=\"status\" aria-hidden=\"true\"></span>" +
        "</button>" +
        "<button class=\"btn\" disabled>\n" +
        "<span class=\"spinner-grow spinner-grow-sm\" role=\"status\" aria-hidden=\"true\"></span>" +
        "</button>\n" +
        "<button class=\"btn\" disabled>\n" +
        "<span class=\"spinner-grow spinner-grow-sm\" role=\"status\" aria-hidden=\"true\"></span>" +
        "</button>\n" +
        "<button class=\"btn\" disabled>\n" +
        "<span class=\"spinner-grow spinner-grow-sm\" role=\"status\" aria-hidden=\"true\"></span>" +
        "</button>";

    const articles = document.getElementById("articles");
    for (const article of Array.from(articles.children)) {
        const ner = article.getElementsByClassName("named-entities")[0];
        ner.innerHTML = nerPlaceholder;
    }
}

