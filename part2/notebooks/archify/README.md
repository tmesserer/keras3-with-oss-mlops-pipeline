# archify

`archify.html` is an optional explainer: the Part 2 data flow told service by service. Nothing in
the repository imports or embeds it, and this whole folder can be deleted without consequence.
`part2/notebooks/pick-me-up.md` is the only file that points at it.

`part2-dataflow.dataflow.json` is the source; the html is the rendering. The page is one
self-contained file, so it keeps working even if the archify skill is gone. To rebuild after
editing the source, from the repository root:

```bash
node .hermes/skills/archify/bin/archify.mjs deliver dataflow \
  part2/notebooks/archify/part2-dataflow.dataflow.json \
  part2/notebooks/archify/archify.html --quality showcase
```

Two facts about the source are worth knowing before editing it. The node grid is fixed — five
stage columns 215px apart, rows 114px apart — so a node's `stage` and `row` are the only placement
there is, and a layout needing a sixth column needs a different diagram. And the page must fit one
screen without scrolling at 1440x900, which is what keeps the picture three rows tall. That height
budget is why `label-sync` and Airflow are named in the cards instead of drawn as boxes.
