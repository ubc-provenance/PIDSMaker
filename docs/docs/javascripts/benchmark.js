// Tooltips of the benchmark charts (docs/scripts/benchmark_charts.py): every mark with a
// data-tip attribute shows its text next to the cursor. Delegated on the document, so it keeps
// working across the instant navigation of the site.
(function () {
  let tip = null;

  function show(target, x, y) {
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "pm-bench-tooltip";
      document.body.appendChild(tip);
    }
    const [head, ...rest] = target.getAttribute("data-tip").split("\n");
    tip.replaceChildren();
    const strong = document.createElement("strong");
    strong.textContent = head;
    tip.appendChild(strong);
    for (const line of rest) {
      const div = document.createElement("div");
      div.textContent = line;
      tip.appendChild(div);
    }
    tip.style.display = "block";
    move(x, y);
  }

  function move(x, y) {
    if (!tip) return;
    const pad = 14;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    const left = x + pad + w > window.innerWidth ? x - pad - w : x + pad;
    const top = y + pad + h > window.innerHeight ? y - pad - h : y + pad;
    tip.style.left = `${Math.max(4, left)}px`;
    tip.style.top = `${Math.max(4, top)}px`;
  }

  function hide() {
    if (tip) tip.style.display = "none";
  }

  document.addEventListener("pointerover", (e) => {
    const target = e.target.closest && e.target.closest(".pm-bench-chart [data-tip]");
    if (target) show(target, e.clientX, e.clientY);
    else hide();
  });
  document.addEventListener("pointermove", (e) => {
    if (tip && tip.style.display === "block") move(e.clientX, e.clientY);
  });
  document.addEventListener("scroll", hide, { passive: true });
})();
