# 💗 Catching Credit Card Fraud with Small Trees

My entry for Kaggle's [Playground Series S6E11: Predicting Credit Card Fraud](https://www.kaggle.com/competitions/playground-series-s6e11) (November 2026), as team **Kar-Joe_Ninjas** with my dad Joe.

**Read the notebook on Kaggle:** [Catching Credit Card Fraud with Small Trees](https://www.kaggle.com/code/jkaren/catching-creditcard-fraud-with-small-trees-kj) (it runs there with all charts and outputs).

## The problem

Each row is a credit card transaction. The task is to give every transaction a fraud score, and Kaggle grades the scores with ROC AUC: the chance that a random fraud gets a higher score than a random normal transaction. Only 1.67% of the 700,000 training transactions are fraud, so "everything is normal" is 98% accurate and completely useless. That's why the score is AUC, not accuracy.

## What I did

I built every model myself, step by step, and wrote down every score. I didn't blend in other people's published predictions: the top of the leaderboard is mostly people averaging the same shared prediction files, and I wanted my score to be my own work.

| Step | What changed | Cross-validation AUC | Public leaderboard |
|---|---|---|---|
| v1 | Logistic regression + LightGBM | 0.94595 | 0.94626 |
| v3 | Target encoding done safely, tiny trees (3–4 leaves), "digit keys" | 0.95045 | 0.95083 |
| v5 | Joint GLM, GLM-boost, train once more on all the data | 0.95044 | 0.95146 |
| v8 | m-estimate encoding, CatBoost, a neural network, hill-climbing blend | **0.95090** | **0.95161** |

Along the way I:

- **Explored the data first:** fraud rates with error bars, risk curves, and an authentication × wrong-CVV heatmap. One wrong CVV attempt raises the fraud rate about 9×.
- **Found a real Simpson's paradox,** where a pattern flips once you compare inside groups.
- **Found clues left by the data generator,** like the cents part of the amount carrying signal that would never exist in real life.
- **Tested PCA** and showed that "most variance" is not the same as "most fraud signal."
- **Showed the leakage trap:** target encoding done the wrong way makes pure random noise look like a great feature.
- **Compared tree sizes** from 2 to 63 leaves. Small trees win here because the signal is mostly additive.
- **Built a straight-line model that rivals the trees** (the joint GLM), then trees that start where it stopped (GLM-boost).
- **Tried a neural network** and reported honestly that it didn't help the blend.
- **Checked how much to trust the leaderboard** with a paired bootstrap. Differences under about 0.0002 are noise.

## My biggest lessons

1. AUC is about order. I can compute it by hand by counting pairs.
2. Accuracy lies when fraud is rare.
3. Error bars keep small groups from fooling me.
4. Always compare inside groups too (Simpson's paradox).
5. Synthetic data has artifacts that real data wouldn't.
6. PCA doesn't know the target.
7. Target encoding is powerful, but it must be cross-fitted.
8. Small trees win when the signal is mostly additive.
9. "Used a lot" isn't the same as "helpful": test a feature by removing it.
10. Trust cross-validation over tiny leaderboard differences.
11. A straight-line model can be strong when you give it great inputs.
12. A blend rewards models that make different mistakes.
13. Training once more on all the data helps, even though CV can't see it.

## What's in this repo

| File | What |
|---|---|
| `notebook/s6e11_showcase.ipynb` | The final notebook (same as on Kaggle) |
| `notebook/build_showcase.py` | Builds the notebook from Python, so every cell is version controlled |
| `notebook/run_local.py` | Runs the notebook on a laptop and saves the charts as images |
| `requirements.txt` | Python packages for running on a laptop |

The competition data isn't included. Download it from the [competition's data page](https://www.kaggle.com/competitions/playground-series-s6e11/data).

**On Kaggle:** add the competition as input and Run All (about 3 hours on CPU).
**On a laptop:** `pip install -r requirements.txt`, then `DATA_ROOT=/path/to/data python notebook/run_local.py` (add `FAST=1` for a quick run on a sample).

## How I worked

My dad Joe was my teammate and mentor. I used Claude, an AI assistant, as a tutor and code helper: it wrote much of the code with me, and I designed the experiments, read every result and decided what to keep. The ideas I learned from other Kagglers are credited inside the notebook.
