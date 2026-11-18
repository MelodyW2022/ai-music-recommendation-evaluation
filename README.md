# Music RAG Reliability Benchmark

## Summary

This project extends my original **Music Recommender Simulation** from the earlier course modules. The original system used an 18-song CSV and a hand-written weighted scoring rule to recommend songs from a user's mood, genre, energy, acousticness, and valence preferences.

The extended version turns that baseline into an applied AI reliability project. It adds offline RAG-style retrieval over a real Spotify-style 500-song sample, grounded explanations, optional Gemini explanation generation, guardrails, deterministic fallback behavior, and an evaluator that measures explanation reliability.

Video walkthrough: https://youtu.be/QBHgzb6XJwI

## What Changed From The Base Project

- Kept the original 18-song mood recommender in `src/recommender.py` as the baseline.
- Added real Spotify-style CSV loading in `src/data_loader.py`.
- Added offline retrieval in `src/retriever.py`; retrieval is the source of truth.
- Added deterministic grounded explanations in `src/explainer.py`.
- Added optional Gemini wording in `src/llm_explainer.py`.
- Added validation guardrails and fallback explanations in `src/guardrails.py`.
- Added reliability metrics in `src/evaluator.py`.
- Added a deterministic 500-row Hugging Face dataset sample in `data/spotify_tracks_sample_500.csv`.

## Architecture

![System Architecture](assets/system_architecture.svg)

Retrieval ranks tracks using genre and audio-feature similarity. In live mode, Gemini writes an explanation, then a separate Gemini request checks each claim against the same track metadata and retrieval evidence. A rejected claim, invalid review, or unavailable judge triggers the deterministic explanation. The offline examples replay simulated generator and judge responses through the same review and fallback logic.

## Dataset

The real-data path uses a deterministic 500-row sample from the Hugging Face Spotify Tracks Dataset by `maharshipandya`. The sample was created with:

```bash
python3 scripts/create_spotify_sample.py
```

The script keeps only the columns required by the loader:

```text
track_name, artists, album_name, track_genre, popularity,
danceability, energy, acousticness, valence, tempo
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Optional Gemini setup:

```bash
export GEMINI_API_KEY="your-key-here"
```

The project still runs without a Gemini key. If the key or SDK is unavailable, it uses deterministic fallback explanations.

## Run The System

Run the end-to-end RAG demo:

```bash
python3 -m src.main demo
```

Show one retrieval and guardrail trace:

```bash
python3 -m src.main trace
```

To generate and judge a live Gemini explanation with an API key:

```bash
python3 -m src.main trace --use-gemini
```

Run reliability evaluation:

```bash
python3 -m src.main evaluate
```

Run tests:

```bash
python3 -m pytest -q
```

Current verified result:

```text
46 passed, 2 skipped
```

With a Gemini API key set, run the two live checks and print the responses:

```bash
RUN_LIVE_GEMINI_TESTS=1 python3 -m pytest -q -s -rs tests/test_llm_explainer.py -k live_gemini
```

One test sends a handwritten claim about the user's listening history through the real judge and requires a rejection and template fallback. The other uses the normal prompt with six recommendation profiles and prints each raw output, its evidence, the judge's findings, and the final text.

## Live Gemini Grounding Results

Both live tests passed on November 18, 2026 with `gemini-3.1-flash-lite`. The tests used *Ishqam*, a pop track with an energy score of `0.84` against the user's target of `0.80`. [GitHub Actions run](https://github.com/MelodyW2022/ai-music-recommendation-evaluation/actions/runs/44421411937/job/135215064365)

### Handwritten hard case

The test supplied this candidate directly to the live Gemini judge:

> It's a pop track, so it should sound a lot like the songs you usually listen to.

The judge rejected the inference about listening history:

> A genre match does not establish a listener's history or confirm that the track sounds like their usual preferences.

The application discarded the candidate and displayed the deterministic explanation from the retrieved track data.

### Normal generation prompt

The live Gemini generator returned:

> This track is a great match because it aligns with your preference for pop music and features a high-energy, danceable electronic sound that closely mirrors your target energy levels.

The live judge accepted the explanation, and the application displayed it as written. This is one example of an accepted output. Estimating the rate of unsupported claims under the normal prompt requires reviewing more outputs against the evidence.

### Six profiles with the normal prompt

A follow-up run used six different preference profiles and retrieved six different tracks. The judge accepted two candidates and rejected four. The following observations come from comparing each candidate with its retrieval evidence. [Raw responses and judge findings](https://github.com/MelodyW2022/ai-music-recommendation-evaluation/actions/runs/44422281225/job/135218030635)

| Track | Judge decision | Evidence review |
| --- | --- | --- |
| Ishqam | Rejected | The pop genre and track tags are supported. The output treats the requested genre as a favorite and suggests the user already enjoys the electronic sound. |
| 2002 - Acoustic | Accepted | The genre matches. Energy rounds to `0.30` against a target of `0.30`, and acousticness is `0.84` against `0.85`. |
| Hot in It | Accepted | The dance genre and track tags are supported. The phrase “electronic sound you enjoy” implies a preference the query does not establish. |
| Santa Cruzin' | Rejected | Jazz matches, but energy `0.64` vs. `0.24` and valence `0.60` vs. `0.36` undercut the claim of a close match. |
| arms | Rejected | The rock genre and track tags are supported. “Favorite” and “vibe you enjoy” go beyond what the current request establishes. |
| Not What I Had In Mind | Rejected | The track is labeled honky-tonk, while the user requested baroque-pop. The candidate correctly identifies the track genre and limits its match to energy (`0.39` vs. `0.36`) and acousticness (`0.54` vs. `0.55`). The judge's rejection cited missing listening history even though the candidate makes no such claim. |

This sample shows both useful rejections and inconsistent judgments about the user's current preferences. The fallback now describes the comparison and presents the evidence without asserting that all audio features closely match.

## Sample Outputs

Demo output shows retrieval over the 500-song sample:

```text
RAG-grounded music recommendation demo
Catalog size: 500
1. Ishqam by Mika Singh;Ali Quli Mirza
   Genre: pop
   Score: 6.31
   Evidence:
   - genre matches preferred genre 'pop' (+2.00)
   - energy closeness: 0.84 vs target 0.80 (+1.92)
```

Trace output shows guardrail fallback:

```text
Gemini mode: simulated explanation and judge response
Raw response: {"explanation": "This track sounds like the songs you usually listen to."}
Used LLM output: False
Fallback reason: Gemini explanation failed grounding check
Guardrail finding: This track sounds like the songs you usually listen to. — The evidence does not include the user's listening history.
Final safe explanation: Ishqam by Mika Singh;Ali Quli Mirza was selected by the recommender. Here is how it compares with your request. ...
```

Evaluation output summarizes reliability:

```text
Evaluation cases: 3
Unsupported claim rate: 0.33
Fallback rate: 0.67
Format failure rate: 0.33
Global evidence coverage: 1.00
```

## Reliability Design

The reliability layer checks that explanations are grounded in retrieval evidence.

Measured metrics:

- `unsupported_claim_rate`: the share of cases whose judge response flags an unsupported or contradicted claim.
- `fallback_rate`: how often the system replaces unsafe or malformed output.
- `format_failure_rate`: how often raw output cannot be parsed as valid JSON.
- `global_evidence_coverage`: the share of expected markers present in the retrieval evidence attached to the final explanations.

By default, the evaluator runs one safe simulated response, one unsupported-claim response, and one malformed response. To run the same retrieval cases with live generation and judging, use `python3 -m src.main evaluate --use-gemini --no-simulated-failures`. Offline metrics describe the scripted outcomes. Live metrics report the judge's decisions. Assessing whether the judge misses claims or rejects supported explanations requires a separately labeled sample.

## Stretch Features

This project includes two stretch-feature candidates:

- **RAG Enhancement:** The original 18-song recommender was extended with offline retrieval over a deterministic 500-song Spotify-style dataset sample. Retrieval produces evidence strings that are used by explanations, guardrails, traces, and evaluator metrics.
- **Test Harness / Evaluation Script:** `src/evaluator.py` and `python3 -m src.main evaluate` run predefined reliability cases and print unsupported-claim rate, fallback rate, format failure rate, and global evidence coverage.

## Design Decisions

- Retrieval is offline and deterministic so recommendations are reproducible.
- Gemini is only a wording layer, not the recommender.
- The original 18-song recommender remains separate from the real-data RAG path.
- The real dataset is sampled deterministically instead of taking the first 500 rows.
- Fallback explanations are deterministic so unsafe or malformed LLM output never breaks the demo.

## Limitations

The system uses metadata and audio features, not lyrics or audio waveforms. It cannot know personal taste, cultural context, vocal style, or lyrical meaning. The 500-song sample is useful for demonstration, but it is still a small slice of a much larger music catalog.

## Reflection

This project taught me that applied AI reliability is mostly about boundaries: deciding what the model is allowed to know, what it must prove with evidence, and what should happen when it fails. The biggest improvement over the baseline is that recommendations now come with retriever evidence and guardrail-checked explanations.

I used AI as a pair-programming partner to design the RAG pipeline, tests, guardrails, and evaluator. A helpful AI suggestion was to keep Gemini as an explanation layer instead of letting it choose recommendations. A flawed suggestion was initially spending time polishing explanation wording before the evaluator and CLI were complete; under the deadline, the higher-value move was to finish the measurable reliability path first.
