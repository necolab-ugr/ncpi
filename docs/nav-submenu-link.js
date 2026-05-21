(function () {
  var labelToFolder = {
    webui: "webui",
    jupyter: "jupyter",
  };

  function resolveIndexUrl(details, folder) {
    var firstSubpageLink = details.querySelector("a.subnav2[href]");
    if (!firstSubpageLink) return null;

    var subpageUrl = new URL(firstSubpageLink.getAttribute("href"), window.location.href);
    var marker = "/" + folder + "/";
    var markerIndex = subpageUrl.pathname.lastIndexOf(marker);
    if (markerIndex === -1) return null;

    var prefixPath = subpageUrl.pathname.slice(0, markerIndex + 1);
    return prefixPath + folder + "-index.html";
  }

  function onSummaryClick(event) {
    var summary = event.currentTarget;
    var label = summary.textContent.trim().toLowerCase();
    var folder = labelToFolder[label];
    if (!folder) return;

    var details = summary.parentElement;
    if (!details || details.tagName !== "DETAILS") return;

    var targetPath = resolveIndexUrl(details, folder);
    if (!targetPath) return;

    var currentPath = window.location.pathname;
    var currentFile = currentPath.slice(currentPath.lastIndexOf("/") + 1);
    var targetFile = targetPath.slice(targetPath.lastIndexOf("/") + 1);

    if (currentFile === targetFile) {
      return;
    }

    event.preventDefault();
    details.open = true;
    window.location.assign(targetPath);
  }

  var summaries = document.querySelectorAll("details.left-nav-collapse > summary.subnav");
  summaries.forEach(function (summary) {
    summary.addEventListener("click", onSummaryClick);
  });
})();
