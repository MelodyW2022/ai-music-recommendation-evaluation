# Music RAG Reliability Benchmark

A Python music recommender that ranks tracks from a reproducible 500-track Spotify sample and evaluates whether AI-generated explanations are supported by retrieval evidence. The earlier 18-song recommender is available in its [original repository](https://github.com/MelodyW2022/ai110-module3show-musicrecommendersimulation-starter).

## Architecture

![Offline retrieval and the single-review explanation flow](assets/system_architecture.svg)

1. **Retrieve and rank.** The Python retriever scores every track against the requested genre and target energy, acousticness, valence, and danceability. It returns the top tracks with their scores and evidence, including the actual and target feature values.
2. **Generate and review.** In live mode, Gemini drafts one explanation from the song name, artist, genre, score, and five scoring comparisons. Claude checks it once when configured. With only a Gemini key, a separate Gemini request performs that check. The reviewer checks each claim, including whether it implies a listening history the request does not provide.
3. **Return the result.** A passing explanation is shown as written. Python builds an explanation from the same evidence when live generation is disabled, the output is unusable, or the review fails. The selected track and ranking come from retrieval.

The catalog is a deterministic sample of the Hugging Face Spotify Tracks Dataset. Regenerate it with `python3 scripts/create_spotify_sample.py`.

[Retriever](src/retriever.py) · [Dataset sample](data/spotify_tracks_sample_500.csv) · [Model card](model_card.md)

## Evaluation

The evaluator tests **explanation reliability** on the first-ranked track for each predefined query. The offline run uses three scripted outcomes to check that a supported explanation passes, an unsupported claim triggers fallback, and malformed JSON triggers fallback.

| Metric | What it measures | Scripted result |
| --- | --- | ---: |
| Unsupported claim rate | Cases where the reviewer flags an unsupported or contradicted claim | 1/3 |
| Fallback rate | Cases where the system substitutes the Python explanation | 2/3 |
| Format failure rate | Cases where the generation cannot be parsed into a valid explanation | 1/3 |
| Final evidence coverage | Expected evidence markers present in final explanations | 100% |

In the original live evaluation, before removing inferred track descriptors, I compared decisions from three reviewers on **eight saved Gemini candidates** against an evidence review. Claude agreed on **8/8**, Gemini on **7/8**, and Groq's GPT-OSS 20B on **3/8**. In the saved *Hot in It* case, Claude caught that the song's `electronic` tag was presented as the listener's preference even though the request specified `dance`. These results describe a small labeled sample.

[Full evaluation report with six profiles, evidence, and model decisions](docs/grounding-evaluation.md) · [Offline evaluator output](assets/evaluation-metrics.svg)

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

python3 -m src.main demo      # Ranked tracks and retrieval evidence
python3 -m src.main trace     # One simulated unsupported claim and its fallback
python3 -m src.main evaluate  # Three scripted reliability cases
python3 -m pytest -q          # Offline test suite
```

For live generation, set `GEMINI_API_KEY`. Also set `ANTHROPIC_API_KEY` to select Claude as the reviewer.

```bash
python3 -m src.main trace --use-gemini
python3 -m src.main evaluate --use-gemini --no-simulated-failures
```

## Tests

The offline suite passed **46 tests**, with five opt-in live tests skipped. It covers retrieval, response parsing, reviewer selection, deterministic fallback, and evaluation metrics. An earlier live integration run confirmed that Claude accepted a freshly generated *2002 - Acoustic* explanation and rejected the saved *Hot in It* candidate, triggering fallback.

[Pytest output](assets/automated-tests.svg) · [Live review excerpt](assets/claude-review.svg) · [Live test commands](docs/grounding-evaluation.md#reproduce-the-live-tests) · [GitHub Actions log](https://github.com/MelodyW2022/ai-music-recommendation-evaluation/actions/runs/44449037925/job/135308148597)

[Original walkthrough](https://youtu.be/QBHgzb6XJwI)
