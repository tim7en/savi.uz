# Documents

Rendered deliverables from the systematic-trading research programme. Each is a
self-contained HTML page — open directly in a browser, no build step and no
network access required.

These are kept here rather than under `out/`, which is gitignored because it
holds regenerable analysis output. A report someone reads is not regenerable
output; losing it to a clean checkout would be losing work.

| File | What it is | Published as |
|---|---|---|
| [`research-agenda.html`](research-agenda.html) | Open experiments, ranked, each with a kill criterion fixed before it runs. The standing controls contract lives here. | [artifact](https://claude.ai/code/artifact/93859ddd-3e9d-40f5-aac9-56a14a675f61) |
| [`report-blueprint.html`](report-blueprint.html) | Chapter-by-chapter plan for the investor report, with the evidence status and closing cost of every gap. | [artifact](https://claude.ai/code/artifact/a8e1c677-b85c-4623-8a7c-c22bca8168e2) |
| [`chapter-02-the-quarter-century.html`](chapter-02-the-quarter-century.html) | Chapter two: sector, commodity, valuation and macro history 2000–2026, measured. | [artifact](https://claude.ai/code/artifact/b27258dc-9d86-4094-a59b-94c68e405447) |

## Regenerating the chapter

Chapter two is generated rather than hand-written, so its figures cannot drift
from the prose that quotes them:

```
python scripts/build_chapter2_data.py     # measures everything into out/report/chapter2.json
python scripts/build_chapter2_page.py     # renders out/report/chapter2.html from the template
cp out/report/chapter2.html docs/chapter-02-the-quarter-century.html
```

The template lives at `scripts/chapter2_template.html`; the charts are inline SVG
built in `build_chapter2_page.py` from the same JSON.

The agenda and blueprint are written by hand and have no generator.

## A note on the numbers

Every performance figure in these documents carries a cost assumption, and it
matters more than any other choice: the banked strategy scores Sharpe 2.64 at a
2bp round trip and 0.25 at 15bp. A figure quoted without its cost is not a
result. The same applies to regime claims, which are reported against the number
of independent *episodes* rather than the number of sessions.
