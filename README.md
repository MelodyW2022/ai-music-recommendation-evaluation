# Music RAG Reliability Benchmark

A Python music recommender that ranks tracks from a reproducible 500-track Spotify sample and evaluates whether AI-generated explanations are supported by retrieval evidence. The earlier 18-song recommender is available in its [original repository](https://github.com/MelodyW2022/ai110-module3show-musicrecommendersimulation-starter).

## Architecture

![Offline retrieval and the single-review explanation flow](assets/system_architecture.svg)

1. **Retrieve and rank.** The Python retriever scores every track against the requested genre and target energy, acousticness, valence, and danceability. It returns the top tracks with their scores and evidence, including the actual and target feature values.
2. **Generate and review.** In live mode, Gemini drafts one explanation from the song name, artist, genre, score, and five scoring comparisons. Claude checks it once when configured. With only a Gemini key, a separate Gemini request performs that check. The reviewer checks each claim, including whether it implies a listening history the request does not provide. Python checks that every Claude claim quote is a verbatim excerpt of the draft.
3. **Return the result.** A passing explanation is shown as written. Python builds an explanation from the same evidence when live generation is disabled, the output is unusable, or the review fails. The selected track and ranking come from retrieval.

The catalog is a deterministic sample of the Hugging Face Spotify Tracks Dataset. Regenerate it with `python3 scripts/create_spotify_sample.py`.

[Retriever](src/retriever.py) · [Dataset sample](data/spotify_tracks_sample_500.csv) · [Model card](model_card.md)

## Evaluation

Run `python3 -m src.main evaluate` to check three preset cases, one top-ranked track per profile. A `1` means the case counts toward that metric.

| Preset case | Unsupported claim | Fallback | Format failure |
| --- | ---: | ---: | ---: |
| Supported explanation is shown | 0 | 0 | 0 |
| Reviewer flags a claim, so the app shows the Python fallback | 1 | 1 | 0 |
| Invalid JSON triggers the Python fallback | 0 | 1 | 1 |
| **Rate across the three cases** | **1/3** | **2/3** | **1/3** |

One case can count in multiple columns. Invalid JSON counts as both a format failure and a fallback.

**Final evidence coverage: 15/15.** Each result carries all five expected scoring evidence fields, for a total of three cases × five fields.

[Detailed evaluation and live results](docs/grounding-evaluation.md) · [Offline evaluator output](assets/evaluation-metrics.svg)

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

python3 -m src.main demo      # Ranked tracks and retrieval evidence
python3 -m src.main trace     # One simulated unsupported claim and its fallback
python3 -m src.main evaluate  # Three preset reliability cases
python3 -m pytest -q          # Offline test suite
```

For live generation, set `GEMINI_API_KEY`. Also set `ANTHROPIC_API_KEY` to select Claude as the reviewer.

```bash
python3 -m src.main trace --use-gemini
python3 -m src.main evaluate --use-gemini --no-simulated-failures
```

## Tests

- **Offline** — Run `python3 -m pytest -q` without API keys. The suite covers ranking, evidence handling, response parsing, reviewer routing, quote validation, fallback, and all four evaluation metrics. The latest run passed **47 tests**.
- **Live** — Opt-in tests exercise generation and review through real APIs, including when the app should accept an explanation or show the fallback.

[Offline test output](assets/automated-tests.svg) · [How to run the live tests](docs/grounding-evaluation.md#reproduce-the-live-tests)

[Original walkthrough](https://youtu.be/QBHgzb6XJwI)
