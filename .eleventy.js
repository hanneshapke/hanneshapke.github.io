module.exports = function (eleventyConfig) {
  // ---- passthrough copy ---------------------------------------------------
  eleventyConfig.addPassthroughCopy("images");
  eleventyConfig.addPassthroughCopy("assets");
  eleventyConfig.addPassthroughCopy("favicon.ico");
  eleventyConfig.addPassthroughCopy("CNAME");

  // ---- watch targets ------------------------------------------------------
  eleventyConfig.addWatchTarget("assets/css/");

  // ---- layout aliases (so existing front matter `layout: post` works) ----
  eleventyConfig.addLayoutAlias("post", "post.njk");
  eleventyConfig.addLayoutAlias("page", "page.njk");
  eleventyConfig.addLayoutAlias("base", "base.njk");

  // ---- markdown -----------------------------------------------------------
  // Allow raw HTML in markdown (posts use <figure>, <iframe>, etc.).
  const markdownIt = require("markdown-it");
  const md = markdownIt({ html: true, linkify: true, typographer: false });
  eleventyConfig.setLibrary("md", md);

  // ---- filters ------------------------------------------------------------
  eleventyConfig.addFilter("isoDate", (d) => new Date(d).toISOString());
  eleventyConfig.addFilter("ymd", (d) =>
    new Date(d).toISOString().slice(0, 10),
  );
  eleventyConfig.addFilter("md", (d) =>
    new Date(d).toISOString().slice(5, 10),
  );
  eleventyConfig.addFilter("year", (d) => new Date(d).getUTCFullYear());

  eleventyConfig.addFilter("wordCount", (content) => {
    if (!content) return 0;
    return content.replace(/<[^>]*>/g, " ").trim().split(/\s+/).filter(Boolean).length;
  });

  eleventyConfig.addFilter("readingTime", (content) => {
    const words = content
      ? content.replace(/<[^>]*>/g, " ").trim().split(/\s+/).filter(Boolean).length
      : 0;
    if (words < 360) return "1 min";
    return Math.ceil(words / 200) + " min";
  });

  eleventyConfig.addFilter("hostOnly", (url) =>
    String(url || "").replace(/^https?:\/\//, "").replace(/\/$/, ""),
  );

  // Group posts by year, newest year first.
  eleventyConfig.addFilter("groupByYear", (posts) => {
    const byYear = {};
    for (const p of posts) {
      const y = new Date(p.date).getUTCFullYear();
      (byYear[y] = byYear[y] || []).push(p);
    }
    return Object.keys(byYear)
      .sort((a, b) => b - a)
      .map((y) => ({ year: y, posts: byYear[y] }));
  });

  // All blog posts, newest first.
  eleventyConfig.addCollection("writing", (api) =>
    api
      .getFilteredByGlob("posts/*.md")
      .sort((a, b) => new Date(b.date) - new Date(a.date)),
  );

  // Group all post tags → [{ tag, posts: [...] }]
  eleventyConfig.addCollection("tagList", (api) => {
    const posts = api.getFilteredByGlob("posts/*.md");
    const tags = {};
    for (const item of posts) {
      const t = item.data.tags;
      if (!t) continue;
      const list = Array.isArray(t) ? t : [t];
      for (const tag of list) {
        (tags[tag] = tags[tag] || []).push(item);
      }
    }
    return Object.keys(tags)
      .sort()
      .map((tag) => ({ tag, posts: tags[tag] }));
  });

  // ---- config -------------------------------------------------------------
  return {
    dir: {
      input: ".",
      output: "_site",
      includes: "_includes",
      data: "_data",
    },
    templateFormats: ["md", "njk", "html", "11ty.js"],
    markdownTemplateEngine: "njk",
    htmlTemplateEngine: "njk",
  };
};
