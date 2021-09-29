let filters = [];
let nerModelID = null;
const baseURL = window.location.origin;

function filterArticles(button) {
    const tag = button.currentTarget;
    const tokens = tag.textContent.trim().split(" ")
    tokens.pop()
    const token = tokens.join(" ")
    if ($(tag).hasClass("active")) {
        $(tag).removeClass("active");
        filters.splice(filters.indexOf(token))
    } else {
        $(tag).addClass("active");
        filters.push(token);
    }
    $.getJSON(baseURL + '/news-filter', {
        'model_id': nerModelID,
        'tags': filters.join('\t'),
    }, function (articles) {
        const articlesNodes = document.getElementById("articles");
        for (const articleNode of articlesNodes.children) {
            const articleID = parseInt(articleNode.id.replace("article-", ""));
            if (!(articles.includes(articleID))) {
                $(articleNode).hide();
            } else {
                $(articleNode).show();
            }
        }
    })
}

$(document).ready(function () {
    nerModelID = document.getElementById("ner-select").value.replace("ner-model-", "");
    $(".single-tag").on('click', filterArticles);
    $("#summarization-select").on("change", function () {
        prepareForNewSummaries();
        const summarizationSelect = document.getElementById("summarization-select");
        const modelID = summarizationSelect.value.replace("summarization-model-", "");

        $.getJSON(baseURL + "/summaries/" + modelID, function (summaries) {
            for (const [articleID, summary] of Object.entries(summaries)) {
                const article = document.getElementById("article-" + articleID);
                const summaryNode = article.getElementsByClassName("summary")[0];
                summaryNode.innerHTML = String(summary);
            }
        })
    });
    $("#ner-select").on("change", function () {
        prepareForNewTagCounts();
        prepareForNewNamedEntities();
        const nerSelect = document.getElementById("ner-select");
        const modelID = nerSelect.value.replace("ner-model-", "");
        filters = [];
        nerModelID = modelID

        $.getJSON(baseURL + '/tags-count/' + modelID, function (tagsCount) {
            let countHTML = "";
            for (const [tagShort, [tagCategory, counts]] of Object.entries(tagsCount)) {
                const tagLower = tagShort.toLowerCase();
                countHTML += "<div class=\"accordion-item\"><h2 class=\"accordion-header\" id=\"show-" + tagLower;
                countHTML += "\"><button class=\"accordion-button btn-light tag-header tag-" + tagLower;
                countHTML += "\" type=\"button\" data-bs-toggle=\"collapse\" data-bs-target=\"#collapse-" + tagLower;
                countHTML += "\" aria-expanded=\"true\">" + tagCategory + "</button></h2><div id=\"collapse-";
                countHTML += tagLower + "\" class=\"accordion-collapse collapse show\" aria-labelledby=\"show-"
                countHTML += tagLower + "\">";
                for (const [token, count] of counts) {
                    countHTML += "<button class=\"btn btn-outline-light tag-" + tagLower + " single-tag\">" +
                        token + " <span class=\"badge bg-secondary\">" + count + "</span></button>"
                }
                countHTML += "</div></div>"
            }
            const tagsMenu = document.getElementById("tags-menu");
            tagsMenu.innerHTML = countHTML;
            $(".single-tag").on('click', filterArticles);
            $.getJSON(baseURL + "/named-entities/" + modelID, function (namedEntities) {
                for (const [articleID, named_entities] of Object.entries(namedEntities)) {
                    let tagHTML = "";
                    for (const [words, entity] of named_entities) {
                        tagHTML += "<div class=\"article-card-entity tag-" + entity.toLowerCase() + " btn\">";
                        tagHTML += words + "<span class=\"badge\">" + entity + "</span></div>";
                    }
                    const article = document.getElementById("article-" + articleID);
                    const nerNode = article.getElementsByClassName("named-entities")[0];
                    nerNode.innerHTML = String(tagHTML);
                }
            })
        });
    });
    $(".open-article").on("click", function (button) {
        const articleID = parseInt(button.currentTarget.id.replace("open-article-", ""));
        $.getJSON(baseURL + "/article-entities/" + articleID + "/" + nerModelID, function (content) {
            $('#article-content-' + articleID).html(content);
        });
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
        console.log(article)
        const summary = article.getElementsByClassName("summary")[0];
        console.log(summary)
        summary.innerHTML = summaryPlaceholder;
    }
}

function prepareForNewTagCounts() {
    const countsPlaceholder =
        "<div class=\"placeholder-wave\">\n" +
        " <span class=\"placeholder col-12 placeholder-lg\"></span>" +
        " <span class=\"placeholder col-12 placeholder-lg\"></span>" +
        " <span class=\"placeholder col-12 placeholder-lg\"></span>" +
        " <span class=\"placeholder col-12 placeholder-lg\"></span>";

    const tagsMenu = document.getElementById("tags-menu");
    tagsMenu.innerHTML = countsPlaceholder;
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

