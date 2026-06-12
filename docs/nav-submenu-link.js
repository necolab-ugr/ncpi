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

  var rightNavLinks = Array.from(document.querySelectorAll(".right-nav a[href^='#']"));
  var rightNavItems = rightNavLinks
    .map(function (link) {
      var id = decodeURIComponent(link.getAttribute("href").slice(1));
      return { link: link, section: document.getElementById(id) };
    })
    .filter(function (item) {
      return item.section;
    });

  if (!rightNavItems.length) return;

  function setActiveRightNav(link) {
    rightNavLinks.forEach(function (candidate) {
      var isActive = candidate === link;
      candidate.classList.toggle("active", isActive);
      if (isActive) {
        candidate.setAttribute("aria-current", "location");
      } else {
        candidate.removeAttribute("aria-current");
      }
    });
  }

  function updateActiveRightNav() {
    // Prefer an explicit hash in the URL (e.g., when clicking a right-nav link)
    if (window.location.hash) {
      var matchingItem = rightNavItems.find(function (item) {
        return item.link.hash === window.location.hash;
      });
      if (matchingItem) {
        setActiveRightNav(matchingItem.link);
        return;
      }
    }

    var visibleItems = rightNavItems.filter(function (item) {
      return !item.section.hidden;
    });
    if (!visibleItems.length) return;

    var activeItem = visibleItems[0];
    var activationLine = Math.min(window.innerHeight * 0.25, 180);

    visibleItems.forEach(function (item) {
      if (item.section.getBoundingClientRect().top <= activationLine) {
        activeItem = item;
      }
    });

    if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) {
      activeItem = visibleItems[visibleItems.length - 1];
    }

    setActiveRightNav(activeItem.link);
  }

  var rightNavUpdatePending = false;
  function scheduleRightNavUpdate() {
    if (rightNavUpdatePending) return;
    rightNavUpdatePending = true;
    window.requestAnimationFrame(function () {
      rightNavUpdatePending = false;
      updateActiveRightNav();
    });
  }

  rightNavItems.forEach(function (item) {
    item.link.addEventListener("click", function () {
      setActiveRightNav(item.link);
    });
  });

  window.addEventListener("scroll", scheduleRightNavUpdate, { passive: true });
  window.addEventListener("resize", scheduleRightNavUpdate);
  window.addEventListener("hashchange", function () {
    var matchingItem = rightNavItems.find(function (item) {
      return item.link.hash === window.location.hash;
    });
    if (matchingItem) setActiveRightNav(matchingItem.link);
    scheduleRightNavUpdate();
  });
  scheduleRightNavUpdate();
})();
