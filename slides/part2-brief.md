# Brief for `slides/part2.html` — handoff to a new session

Working note, not project documentation. It is untracked on purpose: delete it once
`part2.html` exists, and put anything durable in `CLAUDE.md` instead. Read `CLAUDE.md`
("The slides", at the end) first — this file only adds what is not there.

## Status

- `slides/slides.html` (2 slides) and `slides/part1.html` (10 slides) are built, reviewed and
  committed (`b6e6817`). Part 1's notebook was trimmed the same day to stop repeating them.
- `slides/part2.html` does not exist. `slides.html` already links to it, so that link is dead
  until it does.
- Budget: **20 slides**, the second half of a 10/20 split decided by the user.

## How the decks work (already built, do not redesign)

- `deck.css` + `deck.js` are shared by all three decks. No build step, no CDN, classic
  `<script src>` so a deck opened with `file://` behaves like one served over HTTP.
- Every size derives from `--u`, one unit of a fixed 16:9 canvas. Ctrl-P prints one slide per
  landscape page, fully revealed.
- **Navigation is two independent axes.** Up/Down = slide, Left/Right = reveals inside the
  slide. Right on the last reveal does nothing, on purpose. `f` full screen, `d` theme.
- Reveals are `class="frag" data-frag="N"`, numeric, and elements sharing an N appear together.
  Interleave prose and diagram pieces by giving them the same or alternating numbers.
- Every slide carries the footnote *"Slides and project were created using AI, really."*,
  written once by `deck.js`. Do not add it per slide.
- Building blocks already in `deck.css`: `.cols two|three|wide-left|wide-right`, `.card`
  (`.accent`, `.accent-2`, `.accent-3`), `.callout`, `.lead`, `.small`, `.where`, `pre` with
  `.hl` spans, `figure`/`figcaption`. `part1.html` adds a local `.stack`/`.layer` block for its
  layer diagram; add Part 2's own local styles the same way.
- Palette: the project palette — `--c1 #2a78d6`, `--c2 #eb6834`, `--c3 #1baf7a` and neutrals,
  the same slots as both `architecture.svg` files, both notebooks' A2 charts and the Part 2
  dashboard. Part 2's three loops are already ①②③ in `c1`/`c2`/`c3` in its SVG.

## Audience and prose rules

- Readers are data-science or machine-learning students about 30% into their specialisation,
  **with no IT background**, reading English as a second language.
- Define **IT and MLOps** vocabulary at first use, in bold, in one or two sentences. Do **not**
  define ML basics — epoch, dropout, precision/recall are already theirs.
- One idea per sentence, under 25 words. No idioms, no phrasal verbs where a plain verb exists.
  British spelling. Direct address ("you close the loop by hand") is the voice.
- Slides are tighter than the notebook: bullets under about 12 words where possible.

## What Part 1's deck already taught — do not explain again

Container · image · read-only layers and the writable layer · image registry · `CMD` and
environment variables at launch · volumes and the shared `/mlruns` path · training vs inference
as two jobs with opposite needs · run, model version, alias, `@champion` · what MLflow is
(logbook + catalogue) · endpoint, JSON request, Pydantic validation · evaluation after training
vs evaluation after inference · `predictions.jsonl`. Reference these, build on them, never
restate them.

**Part 2's job is the ideas Part 1 could not teach**, in the user's words: automation of the
training loop, labelling, retraining, an application that depends on the endpoint, a versioned
dataset — and it should open the reader's mind to where an LLM would plug into the same loop.
The dataset (card fraud, an analyst reviewing decisions) was chosen because it matches the
Luxembourg finance sector.

## Material to work from

- `part2/notebooks/hub.ipynb` — the scope. Stages A–G, three loops.
- `part2/notebooks/architecture.svg` — the one diagram. **It has no per-stage groups yet.**
  Part 1's got `<g id="dg-…">` / `<g id="fl-…">` wrappers so `part1.html` could reveal it stage
  by stage; do the same for Part 2 (harmless in the notebook, load-bearing for the deck), then
  inline a copy in the deck: drop the file's own `<style>`, prefix its class names `a-`, prefix
  its marker ids, and take colours from `deck.css` so the diagram follows the deck's theme.
  `part1.html` is the worked example.
- Stage codes, from `CLAUDE.md`: A1 fetch · A2 explore · A3 features | B1 train · B2 threshold ·
  B3 promote | C1 endpoint · C2 decision band | D1 consumer · D2 failure handling | E1 review
  queue · E2 verdicts back | F1 build snapshot · F2 hash on the run | G1 retrain DAG · G2
  label-sync DAG.
- Candidate concepts a non-IT reader will stall on, none of them yet covered: class imbalance
  and why accuracy lies at a 0.4% base rate · precision/recall trade-off as an operating
  threshold · a decision band and why a middle band exists · a service depending on another
  service, timeouts and a circuit breaker · a queue and human labelling · label latency ·
  Postgres as shared state · a webhook · data versioning and content hashes · leakage ·
  a DAG and a scheduler · training/serving skew · where an LLM would sit in this loop.

## Working agreement the user expects

1. **The user will supply a slide-by-slide outline for Part 2**, as they did for Part 1's ten
   slides. Take it as the scope, react to it with feedback, and wait for the go before writing
   HTML. Part 1 went through one such round and it is the pattern.
2. Build, then show screenshots of representative slides in both themes before asking for
   approval.
3. After the deck is approved, **trim `part2/notebooks/hub.ipynb`** the way Part 1's was:
   remove the explanation a slide now carries, keep the operational text, add "slide N"
   pointers and a "read the deck first" line at the top.
4. Then update `CLAUDE.md` ("The slides") and commit. The user commits on
   `docs/notebooks-as-documentation`; do not push.

## Rendering screenshots without a display

The Playwright MCP server wants a Chrome that is not installed; the Playwright Chromium is:
`~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome`. Copy the deck to a scratch
directory, append a small script after `deck.js` that dispatches `ArrowDown`/`ArrowRight`
`KeyboardEvent`s from `?s=<slide>&n=<reveals>` query parameters (and sets `data-theme` for
`?theme=light`), then:

```bash
chrome --headless=new --disable-gpu --no-sandbox --hide-scrollbars \
  --force-device-scale-factor=1 --window-size=1440,810 --virtual-time-budget=4000 \
  --screenshot=out.png "file:///path/to/copy/part2.html?s=6&n=4"
```

The MNIST digit grid on Part 1's slide 9 was rendered from `data/mnist.npz` with numpy and
Pillow in `.venv`, then embedded as a data URI. The same approach works for any Part 2 chart —
but load the `dataviz` skill before drawing a real chart.
