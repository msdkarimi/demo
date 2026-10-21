# Failure-flag audit — labelling rules

Sample: 60 conversations drawn at random (seed 11) from the 780 training conversations with
`likely_failure = true`. Labelled by Claude (model) from the whole conversation (first 12 turns),
not a person.

| Label | Meaning |
|---|---|
| failure | The assistant did not give the user what they wanted: the user repeats or rewords the *same* request, corrects it, complains, gives up, or works around a refusal |
| iteration | The user sends new variants of a template (new year / name / slot, "another 20", "continue", "rewrite but …") and the assistant answers each one |
| continuation | A normal multi-turn flow — different follow-up questions, a multi-part task — with no repair |
| unclear | Can't tell (e.g. an identical creative prompt re-sent: regenerate for variety, or dissatisfaction) |

Borderline rules: an identical *factual or task* question re-asked counts as failure (the user
doubted the answer); an identical *creative* prompt re-sent counts as unclear. A conversation with
a failure that was later repaired still counts as failure.
