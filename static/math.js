document.addEventListener("DOMContentLoaded", () => {
  const article = document.querySelector("article");

  if (!article || typeof renderMathInElement !== "function") {
    return;
  }

  renderMathInElement(article, {
    delimiters: [
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
    ],
    throwOnError: false,
  });
});
