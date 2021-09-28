function openArticle(articleID) {
    const baseURL = window.location.origin;
    window.location.href = baseURL + "/article/" + articleID;
}