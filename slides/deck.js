/* Navigation for the three decks. Classic script, no modules, so a deck opened with
   file:// behaves exactly like one served over http.

   Up / Down  — previous / next slide.
   Left / Right — step the reveals inside the current slide. The two axes never mix:
   Right on the last reveal does nothing, so you cannot lose your place while talking. */

(function () {
  var slides = [].slice.call(document.querySelectorAll(".slide"));
  var i = 0, step = 0;
  var theme = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";

  // A reveal step is one or more elements sharing a data-frag number. Steps run in
  // numeric order, which is what lets a bullet and a piece of a diagram appear together.
  // An element without data-frag falls back to its position in the document.
  function stepsOf(slide) {
    var frags = [].slice.call(slide.querySelectorAll(".frag"));
    var auto = 0, keys = [];
    frags.forEach(function (el) {
      var k = el.dataset.frag ? Number(el.dataset.frag) : ++auto;
      if (keys.indexOf(k) < 0) keys.push(k);
      el._key = k;
    });
    keys.sort(function (a, b) { return a - b; });
    slide._frags = frags;
    return keys;
  }
  slides.forEach(function (s) { s._keys = stepsOf(s); });

  var bar = document.createElement("div");
  bar.className = "progress";
  var counter = document.createElement("div");
  counter.className = "counter";
  var note = document.createElement("div");
  note.className = "footnote";
  note.textContent = "Slides and project were created using AI, really.";
  document.body.appendChild(bar);
  document.body.appendChild(counter);
  document.body.appendChild(note);

  function render() {
    slides.forEach(function (s, n) { s.classList.toggle("current", n === i); });
    var s = slides[i];
    s._frags.forEach(function (el) {
      el.classList.toggle("on", s._keys.indexOf(el._key) < step);
    });
    var dots = s._keys.map(function (_, n) { return n < step ? "●" : "○"; }).join("");
    counter.innerHTML = (dots ? '<span class="dots">' + dots + "</span>" : "") +
                        (i + 1) + " / " + slides.length;
    bar.style.width = ((i + 1) / slides.length * 100) + "%";
    if (location.hash !== "#" + (i + 1)) history.replaceState(null, "", "#" + (i + 1));
  }

  function go(n, atEnd) {
    if (n < 0 || n >= slides.length) return;
    i = n;
    step = atEnd ? slides[i]._keys.length : 0;
    render();
  }

  document.addEventListener("keydown", function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    switch (e.key) {
      case "ArrowDown": case "PageDown": case " ": go(i + 1, false); break;
      case "ArrowUp":   case "PageUp":            go(i - 1, true);  break;
      case "ArrowRight": if (step < slides[i]._keys.length) { step++; render(); } break;
      case "ArrowLeft":  if (step > 0)                      { step--; render(); } break;
      case "Home": go(0, false); break;
      case "End":  go(slides.length - 1, true); break;
      case "f": if (document.fullscreenElement) document.exitFullscreen();
                else document.documentElement.requestFullscreen(); break;
      case "d":
        theme = theme === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", theme);
        break;
      default: return;
    }
    e.preventDefault();
  });

  var start = parseInt((location.hash || "").slice(1), 10);
  go(start > 0 && start <= slides.length ? start - 1 : 0, false);

  var hint = document.querySelector(".hint");
  if (hint) setTimeout(function () { hint.style.opacity = "0"; }, 6000);
})();
