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

Retrieval ranks tracks using genre and audio-feature similarity. In live mode, Gemini writes a candidate explanation once. When `ANTHROPIC_API_KEY` is set, Claude Haiku 5.5 checks that candidate against the retrieved evidence. Otherwise, a separate Gemini request checks it. The system displays the candidate when the review passes. If no usable candidate is generated or the reviewer rejects it or cannot return a valid review, the system displays a deterministic explanation based on the retrieved evidence. The ranked recommendation is still returned. Offline examples replay simulated responses through the review and fallback logic.

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

Optional credentials for live mode:

```bash
export GEMINI_API_KEY="your-key-here"
# Enable the second review:
export ANTHROPIC_API_KEY="your-key-here"
```

Without a Gemini key or SDK, the project uses deterministic explanations. With only the Gemini key, live mode uses the Gemini review. With both keys, a live explanation must pass both reviews.

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
50 passed, 5 skipped
```

With a Gemini API key set, run the live checks in separate batches and print the responses:

```bash
RUN_LIVE_GEMINI_TESTS=1 python3 -m pytest -q -s -rs tests/test_llm_explainer.py \
  -k 'live_gemini and not generation_and_review'
sleep 61
RUN_LIVE_GEMINI_TESTS=1 python3 -m pytest -q -s -rs tests/test_llm_explainer.py \
  -k 'live_gemini and generation_and_review'
```

One test sends a handwritten claim about the user's listening history through the real judge and requires a rejection and template fallback. The replay test gives the saved candidates to the judge. The generation test uses the normal prompt with six profiles and prints each raw output, its evidence, the judge's findings, and the final text.

With both keys set, run the live test for the Gemini generation and Claude review path:

```bash
RUN_LIVE_GEMINI_TESTS=1 RUN_LIVE_CLAUDE_REVIEW=1 \
  python3 -m pytest -q -s -rs tests/test_llm_explainer.py -k live_claude_review
```

This test makes a real Gemini generation request followed by a real Claude review. It also sends two saved candidates to Claude. The supported *2002 - Acoustic* explanation should pass. The *Hot in It* explanation should be rejected because it treats the track's electronic tag as a user preference, so the application should display the deterministic explanation. On the draft PR, the `run-claude-closeout` label runs the offline suite and this live check.

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

### Six normal-prompt candidates before the judge update

This is the baseline from the original live run. Each case has the unedited generator text, the original judge decision and any rejection reasons, and a separate evidence review. The case labels were assigned after reviewing the outputs. “Boundary” here concerns the scope or wording of the claim. The rubric did not set a numeric closeness threshold. [Original run and displayed outputs](https://github.com/MelodyW2022/ai-music-recommendation-evaluation/actions/runs/44422281225/job/135218030635)

#### Ishqam

**Case type:** Hard case — unsupported long-term taste

**Gemini candidate**

> Ishqam is a great match for your preferences because it aligns with your favorite pop genre and delivers the high-energy, danceable electronic sound you enjoy.

**Retrieved evidence**

```text
["genre matches preferred genre 'pop' (+2.00)", 'energy closeness: 0.84 vs target 0.80 (+1.92)', 'acousticness closeness: 0.13 vs target 0.20 (+0.93)', 'valence closeness: 0.61 vs target 0.85 (+0.76)', 'danceability closeness: 0.73 vs target 0.80 (+0.70)', 'derived vibe tags for explanation: high_energy, electronic, danceable']
```

**Original judge:** Rejected. The application showed the fallback.

> Ishqam is a great match for your preferences. — The evidence shows a retrieval score and attribute closeness, but does not confirm the user's specific preferences or history.
>
> It delivers the high-energy, danceable electronic sound you enjoy. — While the track has these vibe tags, there is no evidence provided that the user enjoys these specific sounds.

**Evidence review:** Reject. The pop match and track tags are supported. “Favorite” and “sound you enjoy” imply a standing taste that the current request does not establish.

#### 2002 - Acoustic

**Case type:** Positive control — clear pass

**Gemini candidate**

> This track is a great match because it aligns with your preference for acoustic music and closely mirrors your target energy and acousticness levels.

**Retrieved evidence**

```text
["genre matches preferred genre 'acoustic' (+2.00)", 'energy closeness: 0.30 vs target 0.30 (+1.99)', 'acousticness closeness: 0.84 vs target 0.85 (+0.99)', 'valence closeness: 0.61 vs target 0.50 (+0.89)', 'danceability closeness: 0.59 vs target 0.55 (+0.72)', 'derived vibe tags for explanation: low_energy, acoustic, fast_tempo']
```

**Original judge:** Accepted. The application showed the candidate.

**Evidence review:** Accept. The acoustic genre matches. Energy rounds to 0.30 against 0.30, and acousticness is 0.84 against 0.85.

#### Hot in It

**Case type:** Borderline wording — subtle unsupported inference

**Gemini candidate**

> Tiësto and Charli XCX's 'Hot in It' is a great match because it aligns with your preferred dance genre and delivers the high-energy, positive, and danceable electronic sound you enjoy.

**Retrieved evidence**

```text
["genre matches preferred genre 'dance' (+2.00)", 'energy closeness: 0.84 vs target 0.90 (+1.88)', 'acousticness closeness: 0.00 vs target 0.10 (+0.90)', 'valence closeness: 0.72 vs target 0.75 (+0.97)', 'danceability closeness: 0.75 vs target 0.90 (+0.64)', 'derived vibe tags for explanation: high_energy, positive, electronic, danceable']
```

**Original judge:** Accepted. The application showed the candidate.

**Evidence review:** Reject. The dance genre and track tags are supported, but “electronic sound you enjoy” attributes a preference beyond the current request. The original judge let this through.

#### Santa Cruzin'

**Case type:** Hard case — overstated numeric match

**Gemini candidate**

> This track is a strong match because it aligns with your preferred jazz genre and features a moderate musical profile that closely mirrors your target energy and mood preferences.

**Retrieved evidence**

```text
["genre matches preferred genre 'jazz' (+2.00)", 'energy closeness: 0.64 vs target 0.24 (+1.20)', 'acousticness closeness: 0.47 vs target 0.82 (+0.65)', 'valence closeness: 0.60 vs target 0.36 (+0.76)', 'danceability closeness: 0.69 vs target 0.31 (+0.47)', 'derived vibe tags for explanation: moderate_profile']
```

**Original judge:** Rejected. The application showed the fallback.

> The track closely mirrors your target energy and mood preferences. — The evidence shows significant numerical differences between the track's attributes (energy 0.64, valence 0.60) and the target values (energy 0.24, valence 0.36), which contradicts the claim of a 'close' mirror.

**Evidence review:** Reject. Jazz matches, but energy is 0.64 vs. 0.24 and valence is 0.60 vs. 0.36. The candidate calls both close.

#### arms

**Case type:** Hard case — unsupported long-term taste

**Gemini candidate**

> This track is a strong match for your preferences because it aligns with your favorite rock genre and features the high-energy, fast-tempo, and darker electronic vibe you enjoy.

**Retrieved evidence**

```text
["genre matches preferred genre 'rock' (+2.00)", 'energy closeness: 0.76 vs target 0.85 (+1.81)', 'acousticness closeness: 0.11 vs target 0.10 (+0.99)', 'valence closeness: 0.25 vs target 0.26 (+0.99)', 'danceability closeness: 0.46 vs target 0.36 (+0.67)', 'derived vibe tags for explanation: high_energy, darker, electronic, fast_tempo']
```

**Original judge:** Rejected. The application showed the fallback.

> This track is a strong match for your preferences. — The evidence provides a retrieval score, but there is no data regarding the user's specific preferences to confirm this is a 'strong match'.
>
> It features the high-energy, fast-tempo, and darker electronic vibe you enjoy. — While the track has these derived vibe tags, there is no evidence provided that the user enjoys these specific attributes.

**Evidence review:** Reject. Rock and the track tags are supported. The current request does not establish that rock is a favorite or that the user enjoys this particular vibe.

#### Not What I Had In Mind

**Case type:** Boundary case — valid comparison despite a genre mismatch

**Gemini candidate**

> This honky-tonk track is recommended because its low-energy, positive, and danceable vibe aligns closely with your preferred acoustic and energy levels.

**Retrieved evidence**

```text
["genre differs from preferred genre: 'honky-tonk' vs 'baroque-pop' (+0.00)", 'energy closeness: 0.39 vs target 0.36 (+1.94)', 'acousticness closeness: 0.54 vs target 0.55 (+0.99)', 'valence closeness: 0.76 vs target 0.68 (+0.92)', 'danceability closeness: 0.73 vs target 0.63 (+0.67)', 'derived vibe tags for explanation: low_energy, positive, danceable']
```

**Original judge:** Rejected. The application showed the fallback.

> The track is recommended because its vibe aligns with your preferences. — While the evidence shows closeness scores for energy and acousticness, there is no evidence provided regarding the user's actual listening history or specific preferences.

**Evidence review:** Accept on grounding. The candidate calls the track honky-tonk and makes a comparison scoped to energy (0.39 vs. 0.36) and acousticness (0.54 vs. 0.55). Its track tags are supported. It could be clearer about the baroque-pop mismatch. The original judge cited missing listening history, which this explanation does not rely on.

### After the judge began distinguishing request preferences from listening habits

The revised rubric treats the selected genre and feature targets in the current profile as evidence of the current request. A favorite, an established taste, or past listening requires separate evidence. The generator prompt, profile values, and scoring were held fixed. The live run first submitted the exact six baseline candidates to the updated judge, then generated six new candidates with the same profiles. [Replay and fresh generation logs](https://github.com/MelodyW2022/ai-music-recommendation-evaluation/actions/runs/44424439891/job/135225510943)

Earlier iterations revealed a genre-scope misread and a free-tier request limit. The workflow now separates the two batches. [Earlier 429 and rubric results](https://github.com/MelodyW2022/ai-music-recommendation-evaluation/actions/runs/44423740349/job/135223085373)

#### Replay of the original candidates

All six revised decisions matched the evidence review. The two changed decisions are visible in the table.

| Track | Original judge | Revised judge | Revised judge findings |
| --- | --- | --- | --- |
| Ishqam | Rejected | Rejected | it aligns with your favorite pop genre — The evidence confirms 'pop' is the preferred genre for this request, but does not establish it as the user's 'favorite' genre. delivers the high-energy, danceable electronic sound you enjoy — While the track has these vibe tags, the evidence does not establish that the user enjoys this sound. |
| 2002 - Acoustic | Accepted | Accepted | Displayed the candidate. |
| Hot in It | Accepted | Rejected | the sound you enjoy — The evidence does not contain information about the user's personal listening history or established preferences beyond the current request. |
| Santa Cruzin' | Rejected | Rejected | closely mirrors your target energy — The energy value is 0.64, which is significantly higher than the target of 0.24. closely mirrors your target mood preferences — The valence value is 0.60, which is significantly higher than the target of 0.36. |
| arms | Rejected | Rejected | it aligns with your favorite rock genre — The evidence confirms the track is rock and matches the requested genre, but there is no evidence that rock is the user's 'favorite' genre. features the high-energy, fast-tempo, and darker electronic vibe you enjoy — The evidence confirms the track has these vibe tags, but there is no evidence that the user enjoys these specific vibes. |
| Not What I Had In Mind | Rejected | Accepted | Displayed the candidate. |

#### Fresh candidates with the same normal prompt

The retrieved tracks and evidence were the same as in the baseline. Two initial reviews were unavailable and triggered fallback. A targeted replay with the same rubric and a required JSON response schema returned decisions for both. The entries below distinguish the initial behavior from the later review.

#### Fresh output for Ishqam

> Ishqam is a great match because it aligns with your preference for pop music and features a high-energy, danceable electronic sound that closely mirrors your target energy levels.

**Initial judge:** Accepted. The application showed the candidate.

**Evidence review:** Accept. Pop is the requested genre, and energy 0.84 vs. 0.80 supports the stated comparison. The audio descriptions are derived track tags.

#### Fresh output for 2002 - Acoustic

> This track is a great match because it aligns with your preference for acoustic music and features energy and acousticness levels that closely mirror your target preferences.

**Initial judge:** Accepted. The application showed the candidate.

**Evidence review:** Accept. Acoustic matches, with energy about 0.30 vs. 0.30 and acousticness 0.84 vs. 0.85.

#### Fresh output for Hot in It

> Tiësto and Charli XCX's 'Hot in It' is a great match because it aligns with your preference for high-energy, danceable electronic music that mirrors your target valence and energy levels.

**Initial judge:** Review unavailable. The application showed the fallback.

**Evidence review:** Flag the wording about the user's preferences. A dance request and energy/danceability targets do not establish a preference for electronic music. The track's electronic tag is supported. Energy 0.84 vs. 0.90 and valence 0.72 vs. 0.75 support the stated numerical comparisons.

**Follow-up judge:** Accepted the original text. The phrase about a preference for electronic music remains a concern in the evidence review. [Follow-up review logs](https://github.com/MelodyW2022/ai-music-recommendation-evaluation/actions/runs/44425214857/job/135228178552)

#### Fresh output for Santa Cruzin'

> This track is a strong match because it aligns with your preferred jazz genre and features a moderate musical profile that closely mirrors your target energy and mood preferences.

**Initial judge:** Rejected. The application showed the fallback.

> closely mirrors your target energy — The energy value is 0.64, which is significantly higher than the target of 0.24.
>
> closely mirrors your target mood preferences — The valence value is 0.60, which is significantly higher than the target of 0.36.

**Evidence review:** Reject. Jazz matches. Energy 0.64 vs. 0.24 and valence 0.60 vs. 0.36 do not support 'closely mirrors.'

#### Fresh output for arms

> This track is a strong match for your preferences because it aligns with your favorite rock genre and features a high-energy, fast-tempo sound that closely mirrors your target acoustic and valence levels.

**Initial judge:** Review unavailable. The application showed the fallback.

**Evidence review:** Reject. Rock matches and the acousticness and valence gaps are both 0.01. The profile does not establish that rock is the user's favorite.

**Follow-up judge:** Rejected the original text and used the fallback. [Follow-up review logs](https://github.com/MelodyW2022/ai-music-recommendation-evaluation/actions/runs/44425214857/job/135228178552)

> it aligns with your favorite rock genre — The evidence confirms the genre is rock, but there is no evidence that rock is the user's favorite genre.

#### Fresh output for Not What I Had In Mind

> This honky-tonk track is recommended because its low-energy, positive, and danceable vibe aligns closely with your preferred acoustic and energy levels.

**Initial judge:** Accepted. The application showed the candidate.

**Evidence review:** Accept on grounding. The sentence identifies honky-tonk as the track genre and compares the requested acousticness and energy targets. Their gaps are 0.01 and 0.03. The explanation could mention the baroque-pop mismatch more explicitly.


### Live comparison across Gemini, Claude, and Groq

All three providers received the same judge instructions, candidate texts, track metadata, and retrieval evidence. This run used the six frozen baseline candidates and the two saved follow-up candidates. The expected decisions reflect the evidence review for this set. The results motivated using Claude as the reviewer when configured. Groq is used in the comparison.

| Candidate | Expected | Gemini Flash-Lite | Claude Haiku 5.5 | Groq GPT-OSS 20B |
| --- | --- | --- | --- | --- |
| Ishqam | Reject | Reject | Reject | Accept |
| 2002 - Acoustic | Accept | Accept | Accept | Accept |
| Hot in It | Reject | Reject | Reject | Accept |
| Santa Cruzin' | Reject | Reject | Reject | Reject |
| arms | Reject | Reject | Reject | Accept |
| Not What I Had In Mind | Accept | Accept | Accept | Accept |
| Hot in It (follow-up) | Reject | Accept | Reject | Accept |
| arms (follow-up) | Reject | Reject | Reject | Accept |

| Reviewer | Agreement | False accepts | False rejects |
| --- | ---: | ---: | ---: |
| `gemini-3.1-flash-lite` | 7/8 | 1 | 0 |
| `claude-haiku-5-5` | 8/8 | 0 | 0 |
| `openai/gpt-oss-20b` | 3/8 | 5 | 0 |

On the *Hot in It* follow-up, Claude identified that `electronic` was a track tag and that the user's selected genre was `dance`. Gemini and Groq both accepted that candidate. Groq rejected the large energy mismatch on *Santa Cruzin'* but accepted several unsupported preference claims.

The initial direct HTTP request to Groq received a 403. The official Groq SDK completed the comparison after bounded transport retries were enabled for transient rate limits. These retries address API availability. Each case records a single returned verdict per model.

[Gemini and Claude job log](https://github.com/MelodyW2022/ai-music-recommendation-evaluation/actions/runs/44430799485/job/135247369947) · [Completed Groq job log](https://github.com/MelodyW2022/ai-music-recommendation-evaluation/actions/runs/44431747230/job/135250544549)

To reproduce the comparison on the draft PR, apply the `run-judge-comparison` label after setting `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, and `GROQ_API_KEY` in repository Actions Secrets. Use `run-groq-diagnostic` for the same cases against Groq alone. To trigger a label-based run again, remove and reapply the label.

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
