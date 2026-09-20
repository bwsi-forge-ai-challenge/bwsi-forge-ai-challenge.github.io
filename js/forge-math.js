/*
 * F.O.R.G.E. AI Challenge - math rendering
 *
 * Add to any page that contains LaTeX:
 *   <script src="../../../js/forge-math.js"></script>
 *
 * Delimiters:
 *   inline   \( ... \)   or   $ ... $
 *   display  \[ ... \]   or   $$ ... $$
 * Only load this on pages that contain math. A page that also has a price
 * such as "$5" must write it as \$5, or that "$" will start an equation.
 */
window.MathJax = {
  tex: {
    inlineMath: [['\\(', '\\)'], ['$', '$']],
    displayMath: [['\\[', '\\]'], ['$$', '$$']],
    processEscapes: true
  }
};

(function () {
  var s = document.createElement('script');
  s.async = true;
  s.src = 'https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js';
  document.head.appendChild(s);
})();
