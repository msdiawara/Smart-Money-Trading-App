/* Runs before the page paints (a file, because the CSP allows no inline scripts): apply the saved theme so the
   page never flashes the wrong colours, and colour the browser bar to match. "auto" (the default) follows the
   phone's light/dark setting. */
(function () {
  var t = "auto";
  try { t = JSON.parse(localStorage.getItem("smr.theme") || '"auto"'); } catch (e) { /* private mode: follow the system */ }
  if (t === "light" || t === "dark") {
    document.documentElement.setAttribute("data-theme", t);
    var metas = document.querySelectorAll('meta[name="theme-color"]');
    for (var i = 0; i < metas.length; i++) {
      metas[i].setAttribute("content", t === "dark" ? "#0f1213" : "#edf0ef");
      metas[i].removeAttribute("media");
    }
  }
})();
