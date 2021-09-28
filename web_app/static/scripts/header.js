function changeNerModel() {
    const nerSelect = document.getElementById("ner-select");
    console.log(nerSelect.value)
}

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

