# Labelling rules used for `labels_claude.csv`

Labelled by Claude (model), blind: only `first_user_message` was read; `predictions.csv`
and `design.csv` were not opened. Rules applied on top of `categories.md`:

1. Greetings, tests, jokes, riddles and unclear one-liners → Chit-chat & other. A clear request
   on a topic the taxonomy lacks (cooking, travel, shopping, legal, dating, sports facts) → None of these.
2. Factual questions in academic subjects (incl. history, maths, philosophy) and homework/quiz
   questions → Science, health & learning.
3. Questions *about* fiction or fandom that don't ask for fiction to be written → None of these.
4. "Act as X" + a real task → the task (persona as label_2). Persona/agent setup with no task, and
   questions about the model itself → Jailbreak & persona prompts.
5. Code present or requested → Coding. Using/troubleshooting software without code → IT & tech support.
6. Rewriting/proofreading: emails and workplace messages → Business communication; other non-fiction
   → Long-form writing & editing.
7. Translating text → Translation, whatever the content. Non-English messages are labelled by task.
8. A bare visual scene ("walking in the forest of fire") → Image generation prompts.
9. Naming/brainstorming for a business or brand → Marketing. Other naming → None of these.
10. Pasted non-fiction text with no instruction → Long-form writing & editing (noted "R10").
11. Describing invented planets, foods, regimes or worlds → Creative writing & role-play.
12. Workplace letters/emails → Business communication first, topic (e.g. Careers & HR) as label_2.
13. Writing an essay, paper, report or presentation → Long-form writing; short homework answers →
    Science, health & learning, unless the subject has its own category (finance/economics →
    Business, code → Coding, grammar/vocabulary → Translation & languages).
14. Editing fiction → Creative writing (Long-form is non-fiction only).
