// Shared styling for the smi report documents.
//
// Two show-rule entry points, imported by both documents so slides and the
// standalone report share type, color and figure treatment:
//
//   slides-theme(title: ..., subtitle: ..., author: ..., date: ...)
//   report-theme(title: ..., subtitle: ..., author: ..., date: ...)
//
// ASCII only, matching the repo-wide convention enforced by
// scripts/check_ascii.py for *.py and *.md. Write "->", "um", "<=" rather
// than the corresponding Unicode glyphs.

// Palette mirrors report/figures/_figcommon.py COLORS, so text accents and
// figure ink are the same hues.
#let accent = rgb("#2a5d9f")
#let ink = rgb("#1c1c1c")
#let muted = rgb("#6a6a6a")
#let rule-color = rgb("#d0d0d0")
#let highlight = rgb("#7b4ea8")

// Font stacks. Typst emits an "unknown font family" warning for every entry it
// cannot resolve, so list only what is actually installed on the platforms
// that build this: Segoe UI and Arial ship with Windows, and DejaVu Sans Mono
// arrives with matplotlib. On a Linux box without Segoe UI, Typst falls
// through to Arial and then to its own default -- adding speculative Linux
// families here would only add warnings on the machine that does the building.
#let body-font = ("Segoe UI", "Arial")
#let mono-font = ("Consolas", "DejaVu Sans Mono")

// ---------------------------------------------------------------------------
// Slides
// ---------------------------------------------------------------------------

// 16:9 at a comfortable reading size. Typst has no built-in slide class, so a
// "slide" here is just a page with a fixed header rule; using plain pages
// keeps the dependency surface at zero.
#let slide-width = 25.4cm
#let slide-height = 14.29cm

#let slides-theme(
  title: "",
  subtitle: "",
  author: "",
  date: "",
  body,
) = {
  set document(title: title, author: author)
  set page(
    width: slide-width,
    height: slide-height,
    margin: (x: 1.5cm, top: 1.3cm, bottom: 1.1cm),
    footer: context {
      set text(size: 9pt, fill: muted)
      grid(
        columns: (1fr, auto),
        align(left, title),
        align(right, str(counter(page).get().first())),
      )
    },
  )
  // 14 pt on a 25.4 cm page is roughly 28 pt on a projected 16:9 slide -- big
  // enough to read from the back of a room, small enough that a slide with a
  // figure and a takeaway still fits one page. Slides that overflow are a bug:
  // check the render, do not let Typst silently continue onto a second page.
  set text(font: body-font, size: 14pt, fill: ink)
  set par(justify: false, leading: 0.65em, spacing: 0.85em)
  show link: set text(fill: accent)
  show raw: set text(font: mono-font, size: 0.85em)
  show figure.caption: set text(size: 10.5pt, fill: muted)
  show figure: set block(spacing: 0.8em)

  // Slide title: heading level 1 starts a new page.
  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    block(
      below: 0.7em,
      {
        text(size: 21pt, weight: "bold", fill: ink, it.body)
        v(-0.35em)
        line(length: 100%, stroke: 1.2pt + accent)
      },
    )
  }
  show heading.where(level: 2): it => block(
    above: 0.7em,
    below: 0.4em,
    text(size: 16pt, weight: "bold", fill: accent, it.body),
  )

  set list(marker: text(fill: accent)[#sym.bullet], spacing: 0.75em)
  set enum(spacing: 0.75em)

  // Title page
  page(
    margin: (x: 2.4cm, y: 2.4cm),
    footer: none,
    {
      v(1fr)
      text(size: 34pt, weight: "bold", title)
      if subtitle != "" {
        v(0.2em)
        text(size: 20pt, fill: muted, subtitle)
      }
      v(0.8em)
      line(length: 45%, stroke: 1.5pt + accent)
      v(0.6em)
      text(size: 15pt, author)
      if date != "" {
        linebreak()
        text(size: 13pt, fill: muted, date)
      }
      v(1fr)
    },
  )

  body
}

// ---------------------------------------------------------------------------
// Standalone report
// ---------------------------------------------------------------------------

#let report-theme(
  title: "",
  subtitle: "",
  author: "",
  date: "",
  body,
) = {
  set document(title: title, author: author)
  set page(
    paper: "us-letter",
    margin: (x: 2.4cm, y: 2.4cm),
    footer: context {
      set text(size: 9pt, fill: muted)
      align(center, str(counter(page).get().first()))
    },
  )
  set text(font: body-font, size: 10.5pt, fill: ink)
  set par(justify: true, leading: 0.62em, first-line-indent: 0pt, spacing: 0.9em)
  set heading(numbering: "1.1")
  show link: set text(fill: accent)
  show raw: set text(font: mono-font, size: 0.9em)

  show heading.where(level: 1): it => block(
    above: 1.4em,
    below: 0.6em,
    text(size: 15pt, weight: "bold", it),
  )
  show heading.where(level: 2): it => block(
    above: 1.1em,
    below: 0.5em,
    text(size: 12pt, weight: "bold", fill: accent, it),
  )

  align(
    center,
    {
      v(0.5cm)
      text(size: 20pt, weight: "bold", title)
      if subtitle != "" {
        linebreak()
        v(0.2em)
        text(size: 12pt, fill: muted, subtitle)
      }
      v(0.5em)
      text(size: 11pt, author)
      if date != "" {
        linebreak()
        text(size: 10pt, fill: muted, date)
      }
      v(0.4cm)
      line(length: 100%, stroke: 0.8pt + rule-color)
    },
  )

  body
}

// ---------------------------------------------------------------------------
// Shared elements
// ---------------------------------------------------------------------------

// Include a generated figure by its Snakefile registry name. The
// build/figures/ prefix and .pdf suffix are applied here so a document names a
// figure exactly as the Snakefile's FIGURES list does.
//
// The leading "/" makes this root-relative rather than relative to this file.
// Typst resolves a relative path against the file the call textually lives in,
// which is this one, in lib/ -- so a bare "build/..." would look for
// lib/build/. `typst compile --root .` is run from report/, so "/build/..."
// is report/build/... regardless of which document imports this.
#let figpath(name) = "/build/figures/" + name + ".pdf"

#let fig(name, caption: none, width: 92%) = figure(
  image(figpath(name), width: width),
  caption: caption,
)

// A short call-out for a claim the reader should carry away from a slide.
#let takeaway(body) = block(
  fill: rgb("#f2f5fa"),
  stroke: (left: 3pt + accent),
  inset: (x: 12pt, y: 9pt),
  radius: 2pt,
  width: 100%,
  body,
)

// Placeholder marker for content a later phase fills in. Deliberately loud,
// so an unfinished section cannot be mistaken for a finished one.
#let todo(body) = block(
  fill: rgb("#fff4e0"),
  stroke: (left: 3pt + rgb("#d99a1c")),
  inset: (x: 12pt, y: 9pt),
  radius: 2pt,
  width: 100%,
  [#text(weight: "bold")[Placeholder. ] #body],
)
