// Shared styling for the smi report documents.
//
// One show-rule entry point, imported by the standalone report for its type,
// color and figure treatment:
//
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

// A short call-out for a claim the reader should carry away.
#let takeaway(body) = block(
  fill: rgb("#f2f5fa"),
  stroke: (left: 3pt + accent),
  inset: (x: 12pt, y: 9pt),
  radius: 2pt,
  width: 100%,
  body,
)
