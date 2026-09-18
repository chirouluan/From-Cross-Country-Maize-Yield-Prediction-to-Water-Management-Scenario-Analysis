# Blinded agronomist review protocol

## Objective

Compare a deterministic evidence template, Base Qwen, and structured-RAG Qwen without showing the
system identity to reviewers.

## Recommended panel

- At least three independent reviewers with crop agronomy, agrometeorology, irrigation,
  or extension experience.
- Reviewers must not participate in system development.
- Record expertise category and years of experience separately from the blinded ratings.
- Follow the institution's determination on consent or ethics review before recruitment.

## Materials and randomization

Export advice from the `deterministic_policy`, `qwen_base`, and `qwen_rag` branches,
replace system names with randomized item IDs, and give reviewers only the blinded
items plus a rating template. Keep the randomization key private until all ratings
are locked.

## Rating dimensions

Each item receives an integer score from 1 (poor) to 5 (excellent):

- Relevance to the supplied evidence
- Practical actionability under the stated limitations
- Safety and appropriate uncertainty language
- Evidence grounding
- Overreach, where 1 means no overreach and 5 means severe overreach

Reviewers may add free-text comments. Missing ratings are not automatically imputed.

## Analysis

Report reviewer-level and item-level distributions, system means with uncertainty,
inter-reviewer agreement, and paired system contrasts. Preserve the blind until exclusions
and scoring rules are finalized. The project must not manufacture placeholder experts or
replace professional judgment with another LLM.
