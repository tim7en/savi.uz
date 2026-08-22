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
| [`chapter-06-the-research-department.html`](chapter-06-the-research-department.html) | Chapter six: delayed 13F filings fail as entries, but concentrated-manager conviction changes the right tail after breakouts. | Local draft |
| [`chapter-07-when-six-slots-are-scarce.html`](chapter-07-when-six-slots-are-scarce.html) | Chapter seven: 13F conviction as a six-slot Turtle allocation priority and a candidate risk tilt. | Local draft |
| [`13f-the-filing-and-the-slot.html`](13f-the-filing-and-the-slot.html) | 13F conviction tested as an entry gate and as a slot priority. | [artifact](https://claude.ai/code/artifact/3f6dccbe-cded-4b6c-8e32-dc6163ccce77) |
| [`where-the-bar-closes.html`](where-the-bar-closes.html) | Close location within the breakout bar as an entry filter. | [artifact](https://claude.ai/code/artifact/be73089e-ed78-47b8-b920-6f4f9b83f4f9) |
| [`the-interval-was-the-parameter.html`](the-interval-was-the-parameter.html) | Seven bar sizes at true cost: the banked 30-minute interval is a trough, and dip buying beats the breakout. | [artifact](https://claude.ai/code/artifact/607cd415-38c6-4bf6-a0dc-edbdf751958d) |
| [`six-proposals-one-survivor.html`](six-proposals-one-survivor.html) | Trade-selection ML, a leveraged 3:1 scalp, support/resistance zones, 20x leverage, the volatility surprise, and option activity — all six die on their controls once the sample is large enough to tell. | [artifact](https://claude.ai/code/artifact/4e10691c-e699-41ad-8ee6-fe934f1675db) |

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

## Chapter Seven data and regeneration

The Chapter Seven data is locally cached under `data/13f/` and intentionally
gitignored because it contains a large vendor-derived daily-bar database:

- `holdings_major.json`: original 13F holdings with public filing dates;
- `h13f.pkl`: holdings matched to tradable tickers;
- `book13f.pkl`: position-level new-position and conviction-weight flags;
- `alphavantage_daily.db`: split-consistent Alpha Vantage daily OHLCV for the
	143 usable high-conviction names. Alpha Vantage rejected `IAC` as an invalid
	symbol, so it is explicitly omitted.

Rebuild the measured result and its rendered chapter with:

```
PYTHONPATH=src python scripts/run_13f_turtle_pilot.py \
	--out out/strategy/13f_turtle_full_panel.json
PYTHONPATH=src python scripts/build_chapter7_13f_turtle.py
```

The chapter is generated from the result JSON. It holds the panel, entry,
exit, cost, and six-position cap constant when comparing conviction priority
and risk tilts; its source text also names the hindsight-universe and
exit-marked-drawdown limits.

## A note on the numbers

Every performance figure in these documents carries a cost assumption, and it
matters more than any other choice: the banked strategy scores Sharpe 2.64 at a
2bp round trip and 0.25 at 15bp. A figure quoted without its cost is not a
result. The same applies to regime claims, which are reported against the number
of independent *episodes* rather than the number of sessions.
