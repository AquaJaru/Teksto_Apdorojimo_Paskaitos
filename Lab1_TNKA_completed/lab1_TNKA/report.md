# Laboratory 1 — Editor's recommendation

**Name:** Akvilė Jarusevičiūtė 
**Date:** 2026-09-24  

**Run instructions, package versions and saved parameters:** see `README.md`. The checked run used pandas 2.2.3, scikit-learn 1.8.0, joblib 1.5.3 and numpy 2.3.5. The fitted selected model is saved as `models/run_B.joblib`; `inference.py` uses that model and its fitted vocabulary without retraining.

> **Checkpoint note:** the course requires the initial interpretation to be completed without AI and the checkpoint file to remain unchanged. `initial-analysis.md` therefore contains the recorded pre-model selections, but the student must complete/verify the required ambiguity explanations and two predictions from their own checkpoint work before submission.

## 1. Decisions and prediction

Two topic counts were compared: **10 topics (Run A)** and **4 topics (Run B)**. The initial sample contains several recurring subject areas—technology/AI, sport, economy/markets and clickbait-like templates—and also contains English/Lithuanian variants of similar headlines. Four topics were therefore plausible as a compact editor-facing organisation, while ten topics provided a useful contrast showing whether the model would split the collection into more specific but potentially fragmented themes.

The preprocessing decision tested was removal of five recurring English filler phrases: `you won't believe`, `what happens next`, `will shock you`, `this trick will change your life`, and `number 5 is shocking`. The prediction was that removing these phrases would reduce the influence of clickbait templates and make topic words more content-specific. The cost predicted was that some very short headlines would lose useful retained terms and become less informative. The test therefore focused especially on template-like headlines such as H01 and H08.

## 2. Three-run comparison

| Run | Topic count | Preprocessing | Fixed settings / seed | Evidence of usefulness or problems |
|---|---:|---|---|---|
| A | 10 | Original | sample seed 42; model seed 42; max_iter 30; min_df 2; max_features 3000 | More specialised topics appear, but several are fragmented or contaminated by unrelated content. Topic 2 combines clickbait terms with sports wording; topic 8 combines open-source/ML terms with unrelated contract/earnings headlines. |
| B | 4 | Original | Same as A; only topic count changed | Broader themes are easier to describe, and the 20-headline outputs are generally concentrated in one topic, but topics still mix semantic areas. Topic 0 mixes markets with sports-contract wording; topic 2 mixes ML with clickbait/sport. |
| C | 4 | Remove five filler phrases | Same as B; only preprocessing changed | Some template words are reduced, but the change also removes signal from short clickbait headlines. H08 falls to only 2 retained terms and becomes nearly split between topics 1 and 3 (0.417/0.416). |

### Concrete preprocessing effect

For H01, the original text `This trick will change your life forever (Kaunas)` has 5 retained terms in Run B and receives 0.872 weight in one topic. After filler removal in Run C it has only 2 retained terms and its dominant topic weight falls to 0.743. This supports the predicted reduction of template influence, but also shows a loss of lexical evidence.

H08 shows the cost more strongly. In Run B, `She did X — what happens next will shock you — Read more` has 4 retained terms and a clear dominant topic with weight 0.850. In Run C, after removing the recurring filler phrases, only 2 terms remain and the weights are almost tied (0.417 and 0.416). Thus the preprocessing decision does not uniformly improve interpretability: it can make short headlines harder to assign.

## 3. Final model and failure analysis

The selected model for the editor-facing recommendation is **Run B: 4 topics with the original preprocessing**. It was selected because the 10-topic model was more fragmented, while the filler-removal experiment showed a concrete cost for short template-like headlines. This is an exploratory teaching-dataset conclusion, not evidence of generalisation to unseen news.

### Comparison with the initial 20

- **H04** (`Treneris aiškino naują gynybos taktiką po rungtynių`) was initially recorded as sport. Run B assigns it strongly to topic 0 (0.893), whose words include market terms but also `gynybos` and `rungtynių`. The dominant assignment is understandable from the sports words, but the topic itself is semantically mixed.
- **H09** (Apple/iPhone/AI) was initially recorded as technology. Run B assigns it strongly to topic 1 (0.924). This agrees with the technology interpretation, although topic 1 also contains security and clickbait vocabulary.
- **H10** (stock market/inflation) was initially recorded as economy. Run B assigns it to topic 0 with weight 0.915. This is strong agreement and is one of the clearer examples in the model.
- **H12** (`10 reasons you should never buy this gadget`) was initially recorded as technology with a note that it could also be clickbait. Run B assigns it to topic 2 (0.812), illustrating that the model responds strongly to the template/gadget wording rather than cleanly separating subject matter from presentation style.

### Three concrete failure cases

| Headline ID and text | Actual model output | Why it fails the editor's needs | Possible remedy or inherent ambiguity |
|---|---|---|---|
| **H15** — `Žvaigždė pasirašė daugiametę sutartį su klubu as markets react (Madrid)` | Topic 0, 0.916 | The headline contains a sports transfer event plus a market-reaction phrase. Topic 0 also contains inflation/stock-market vocabulary, so the assignment is influenced by mixed signals rather than representing one clean subject. | Multi-topic display or human review could preserve both sport and economy interpretations. |
| **H07** — `Atviro kodo bibliotekos pagreitina mašininio mokymosi procesus` | Topic 2, 0.906 | The dominant topic contains useful ML/open-source words, but its top words also include `messi`, `life`, `trick` and other unrelated/template vocabulary. The topic therefore does not form a clean technology cluster. | Increase corpus diversity and improve preprocessing; alternatively allow a headline to be represented by multiple topics instead of using only the dominant topic. |
| **H12** — `10 reasons you should never buy this gadget` | Topic 2, 0.812 | The headline is simultaneously about a gadget and written as a clickbait-style list. A single topic cannot clearly express whether the editor should treat it as technology content or a presentation/template type. | Keep content and template features separate, or use human judgement for inherently multi-aspect headlines. |

These represent at least two distinct problems: **mixed-topic headlines** (H15/H12) and **semantically contaminated/broad topics** (H07). The results also show why topic IDs should be interpreted through their words and examples rather than their numeric labels.

## 4. Recommendation

Use **4 topics with the original preprocessing** as the provisional editor-facing model. The 10-topic run creates more fragmented themes, while the filler-removal run demonstrates that removing recurring phrases can reduce useful evidence in short headlines. The accepted trade-off is that the four topics remain broad and sometimes mix subjects such as sport, markets, technology and clickbait language.

The model should therefore be treated as an exploratory organisation aid rather than an automatic final classifier. A human editor is still needed for headlines containing more than one subject, repeated templates, mixed languages, or insufficient lexical evidence. The teaching collection itself contains repeated/template-like and English/Lithuanian variants, so these results should not be presented as generalisation to real-world news.

## 5. Sources and AI use

**Source:** scikit-learn documentation for `LatentDirichletAllocation` and the library implementation used in the code. The assignment also specifies pandas/scikit-learn as permitted tools.

**AI use:** AI assistance was used after the initial-analysis checkpoint to inspect the submitted implementation/results, identify missing report requirements, structure the report, and prepare documentation/inference support. The generated text was checked against the submitted CSV outputs and code; numerical claims in this report come from those files. One concrete suggestion that was verified was to compare the preprocessing change using the same topic count and model seed; the existing Run C does this. The initial interpretation/checkpoint and practical defence must remain the student's own work without AI assistance, as required by the assignment.

## Defence preparation

Use `inference.py` with the saved Run B model. It transforms new headlines using the already fitted vectorizer and LDA model. If no retained vocabulary terms remain, it reports that there is insufficient evidence for a meaningful assignment instead of inventing a topic.

During the defence, first predict a likely topic and uncertainty from the headline, then run the saved model and explain any difference.
