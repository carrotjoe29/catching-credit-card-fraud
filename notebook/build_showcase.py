"""Build Karen's public showcase notebook: notebook/s6e11_showcase.ipynb

v5 (final, public): + joint GLM, GLM-boost, neural network, full refit, hill-climbing blend.
v4: Markdown headings for every section and subsection, so Kaggle builds the Table of Contents.
v3: 'My teammate and dad <b>Joe</b>' in the section-3 callout (Joe's edit, as published).
v2: section 11 separates "noise of one score" from "noise of the difference between two
submissions" (paired bootstrap); experiment log gains the v7 row.
Usage: python notebook/build_showcase.py
All charts use one validated palette (purple = normal, pink = fraud, teal = third series,
RdPu for heatmaps). Narrative is first person (Karen). Numbers in the text are either
printed by the code itself or come from our logged experiments.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n")}


def note(title, body, emoji="📓"):
    """A soft pink callout box (Kaggle renders inline HTML in markdown)."""
    return md(f'''
<div style="background:#fdf0f7; border-left:6px solid #b04fc8; border-radius:10px; padding:14px 18px; margin:6px 0;">
<b style="color:#7b3fbf;">{emoji} {title}</b><br>
{body}
</div>
''')


def section(num, title, subtitle):
    """A Markdown heading (so Kaggle's Table of Contents lists it) + a pink banner with the subtitle."""
    return md(f'''
# {num}. {title}
<div style="background:linear-gradient(90deg,#f7d6ea,#e6d3f7); border-radius:14px; padding:10px 20px;">
<span style="color:#5a2a86; font-size:1.05em;">🌸 {subtitle}</span>
</div>
''')


CELLS = [
    md('''
# 💗 Catching Credit Card Fraud with Small Trees 🌳
<div style="background:linear-gradient(120deg,#fbd3e9 0%,#e3c8f5 55%,#cfe3f7 100%); border-radius:18px; padding:28px 30px; text-align:center;">
<h3 style="color:#7b3fbf; margin-top:0;">My learning journal for Playground S6E11 — EDA, honest testing, and what actually helped</h3>
<p style="color:#6b5a73;">by <b>Karen</b> · team <i>Kar-Joe_Ninjas</i></p>
</div>
'''),
    md('''
Hi! 👋 I'm Karen, and this is my notebook for the **Predicting Credit Card Fraud** Playground competition.

I didn't want to just post a score. I wanted to write down **everything I learned** while working on this
competition, in the order I learned it — the charts that surprised me, the mistakes I almost made, and the
experiments that did *and didn't* work. If you're also learning, I hope this helps you as much as writing it
helped me. 💜

**What's inside**

| # | Section | What you'll see |
|---|---|---|
| 1 | The problem & the score | What ROC AUC means — worked out **by hand** first |
| 2 | First look at the data | Size, fraud rate, column types |
| 3 | EDA: where does fraud hide? | Merchant types, login methods, risk curves, a heatmap |
| 4 | A real Simpson's paradox | When a pattern flips inside groups |
| 5 | Clues left by the data generator | Why the *digits* of a number matter here |
| 6 | Does PCA find fraud? | Variance vs. fraud signal |
| 7 | Target encoding — and the leakage trap | The same column scores 0.5 or "amazing" depending on how you encode it |
| 8 | Why small trees win | 2 leaves vs. 63 leaves, measured |
| 9 | My first three models: Logistic, LightGBM, CatBoost | 5-fold CV and a first blend |
| 10 | A smarter straight-line model: the joint GLM | Logistic regression with much better inputs |
| 11 | Trees that start where the GLM stopped | "GLM-boost" — the trees only learn the GLM's mistakes |
| 12 | A neural network | Smooth curves instead of steps — does it help? |
| 13 | Train once more on everything | The "full refit" trick for the final predictions |
| 14 | The final blend: hill climbing | Letting the data pick the mix of models |
| 15 | Used a lot ≠ actually helpful | Feature importance vs. removing columns |
| 16 | How much should I trust the leaderboard? | The math of leaderboard noise |
| 17 | My experiment log | Every version I tried, with honest scores |
| 18 | What I learned & thank-yous | |

<i>⏱️ This notebook trains a lot of models — it takes about 3 hours on Kaggle's CPU. All the
results below are from the saved run.</i>

<i>Colours: in every chart, <b style="color:#e0569b;">pink = fraud</b> and <b style="color:#7b3fbf;">purple = normal</b>.
I checked that these two colours stay easy to tell apart for people with colour blindness too.</i>
'''),
    code(r'''
# 🛠️ Setup: libraries, my colour palette, and the data
import os, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")

# My palette (checked for colour-blind safety): purple = normal, pink = fraud, teal = a third thing
PURPLE, PINK, TEAL = "#7b3fbf", "#e0569b", "#2a8fb8"
LAVENDER = "#d9c6ef"                       # soft colour for "everything else" bars
INK, SOFT_INK, GRID, BG = "#2d1b33", "#6b5a73", "#efe3f3", "#fffafd"
SEQ = "RdPu"                               # light pink -> deep purple, for heatmaps
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": SOFT_INK, "ytick.color": SOFT_INK, "axes.titleweight": "bold",
    "axes.titlesize": 13, "axes.titlecolor": "#4a1f73", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "legend.frameon": False, "figure.dpi": 110, "lines.linewidth": 2, "axes.axisbelow": True,
})

FAST = os.environ.get("FAST", "0") == "1"   # FAST=1 uses a sample, for trying the notebook quickly

def find_data(root):
    """Find the folder with train.csv and test.csv (Kaggle puts it under /kaggle/input/competitions/...)."""
    for p in sorted(Path(root).rglob("train.csv")):
        if (p.parent / "test.csv").exists():
            return p.parent
    raise FileNotFoundError(f"no train.csv + test.csv under {root}")

DATA = find_data(os.environ.get("DATA_ROOT", "/kaggle/input"))
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
if FAST:
    train = train.sample(min(len(train), 120_000), random_state=0).reset_index(drop=True)

TARGET, ID = "is_fraud", "id"
y = train[TARGET].values
FEATS = [c for c in train.columns if c not in (ID, TARGET)]
FLAGS = [c for c in FEATS if train[c].dtype == bool]
CAT = [c for c in FEATS if c not in FLAGS and not pd.api.types.is_numeric_dtype(train[c])]   # word columns
NUM = [c for c in FEATS if c not in FLAGS + CAT]
FLOATS = [c for c in NUM if train[c].dtype.kind == "f"]
T0 = time.time()
print(f"data folder: {DATA}")
print(f"train {train.shape}, test {test.shape}, FAST={FAST}")
'''),

    # ---------------------------------------------------------------- 1
    section(1, "The problem & the score", "What are we predicting, and how is it graded?"),
    md('''
Every row is one credit card transaction. For each transaction in the test set we predict the
**probability that it is fraud** (`is_fraud`).

The score is **ROC AUC**. When I first read that, it sounded scary, so I looked for the simplest way to
say it:

> **AUC = if I pick one random fraud and one random normal transaction, how often does my model give the
> fraud the higher score?**

- AUC **1.0** → the model always ranks fraud above normal (perfect)
- AUC **0.5** → a coin flip
- Ties count as half a win

That means only the **order** of my predictions matters, not the exact numbers. I tested this by hand
before believing it. ✏️
'''),
    note("By hand first ✏️",
         "2 frauds with scores <b>0.9</b> and <b>0.4</b>, and 3 normal transactions with <b>0.5, 0.2, 0.1</b>.<br>"
         "That makes 2 × 3 = <b>6 pairs</b>. The fraud at 0.9 beats all three. The fraud at 0.4 beats 0.2 and 0.1 "
         "but loses to 0.5. So the fraud wins <b>5 of 6</b> pairs → AUC = 5/6 ≈ <b>0.833</b>. Let's check with code:"),
    code(r'''
toy_y = np.array([1, 1, 0, 0, 0])
toy_p = np.array([0.9, 0.4, 0.5, 0.2, 0.1])
rows = []
for i in np.where(toy_y == 1)[0]:
    for j in np.where(toy_y == 0)[0]:
        win = 1.0 if toy_p[i] > toy_p[j] else 0.5 if toy_p[i] == toy_p[j] else 0.0
        rows.append({"fraud score": toy_p[i], "normal score": toy_p[j], "fraud wins?": win})
pairs = pd.DataFrame(rows)
display(pairs)
print(f"by hand: {pairs['fraud wins?'].sum():.0f} wins / {len(pairs)} pairs = {pairs['fraud wins?'].mean():.3f}")
print(f"sklearn roc_auc_score:          {roc_auc_score(toy_y, toy_p):.3f}")
print(f"double every score -> AUC is    {roc_auc_score(toy_y, toy_p * 2):.3f}   (only the ORDER matters!)")
'''),

    # ---------------------------------------------------------------- 2
    section(2, "First look at the data", "How big is it, how rare is fraud, what kinds of columns?"),
    code(r'''
print(f"rows in train: {len(train):,}   rows in test: {len(test):,}")
print(f"fraud rate in train: {y.mean():.2%}  ->  about 1 in {1 / y.mean():.0f} transactions")
print(f"\n{len(NUM)} number columns : {NUM}")
print(f"{len(CAT)} word columns   : {CAT}")
print(f"{len(FLAGS)} yes/no flags   : {FLAGS}")
print(f"\nmissing values: {int(train.isna().sum().sum())}   duplicate rows: {int(train.duplicated(subset=FEATS).sum())}")
train.head()
'''),
    note("Why 'accuracy' would fool me 🙈",
         "Fraud is rare. A model that <i>always</i> says \"not fraud\" would be right almost every time and "
         "score about 98% accuracy — while catching <b>zero</b> fraud. That's why this competition uses AUC, "
         "which only cares whether frauds are ranked above normal transactions."),

    # ---------------------------------------------------------------- 3
    section(3, "EDA: where does fraud hide?", "Fraud rate by group, with error bars so small groups don't fool me"),
    md('''
For every chart in this section I show the **fraud rate** of each group with a **95% Wilson interval**
(the little error bar). I learned to add these because a category with only a handful of transactions can
look super risky just by luck — the error bar shows how sure we can be.

The dashed purple line is the overall fraud rate, so anything to the right of it is riskier than average.
'''),
    code(r'''
def wilson(k, n, z=1.96):
    """95% Wilson interval for a rate k/n (works well even when the rate is tiny)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half

def rate_table(col, df=train):
    g = df.groupby(col, observed=True)[TARGET].agg(frauds="sum", n="size")
    g["rate"] = g.frauds / g.n
    g["lo"], g["hi"] = wilson(g.frauds, g.n)
    return g.sort_values("rate")

def rate_bars(ax, col):
    t = rate_table(col)
    pos = np.arange(len(t))
    ax.barh(pos, t.rate, color=PINK, height=0.6)
    ax.errorbar(t.rate, pos, xerr=[t.rate - t.lo, t.hi - t.rate], fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    ax.axvline(y.mean(), color=PURPLE, ls="--", lw=1.5)
    ax.set_yticks(pos, t.index)
    for p_, r, h, n in zip(pos, t.rate, t.hi, t.n):
        ax.text(h + t.hi.max() * 0.03, p_, f"{r:.1%}  (n={n:,})", va="center", fontsize=9, color=SOFT_INK)
    ax.set_xlim(0, t.hi.max() * 1.55)
    ax.set_title(f"Fraud rate by {col}")
    ax.set_xlabel("fraud rate")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
rate_bars(axes[0], "merchant_category")
rate_bars(axes[1], "auth_method")
plt.tight_layout(); plt.show()
'''),
    code(r'''
fig, axes = plt.subplots(1, 3, figsize=(17, 4))
for ax, col in zip(axes, ["channel", "device_type", "card_type"]):
    rate_bars(ax, col)
plt.tight_layout(); plt.show()
'''),
    note("What I noticed 🔍",
         "Some merchant types and some ways of logging in (authentication methods) are far riskier than others, "
         "and their error bars don't overlap with the safe ones — so it's not luck. My teammate and dad <b>Joe</b> works on fraud "
         "prevention, and he told me this matches real life: gift cards and crypto are easy to turn into cash, "
         "and a purchase with no authentication is easier for a thief."),
    md('''
## How strong is each number column *on its own*?

Before building any model, I asked: if I sorted all transactions by just **one** column, what AUC would I
get? (0.5 = that column alone tells you nothing.) If a column works "backwards" (lower = riskier), I flip it,
so every column gets a fair score.
'''),
    code(r'''
solo = pd.Series({c: max(roc_auc_score(y, train[c]), 1 - roc_auc_score(y, train[c])) for c in NUM}).sort_values()
fig, ax = plt.subplots(figsize=(9, 5.5))
colors = [PINK if v >= solo.sort_values().iloc[-4] else LAVENDER for v in solo]
ax.barh(solo.index, solo - 0.5, left=0.5, color=colors, height=0.6)
for i, v in enumerate(solo):
    ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9, color=SOFT_INK)
ax.axvline(0.5, color=PURPLE, ls="--", lw=1.5)
ax.set_xlim(0.49, solo.max() + 0.04)
ax.set_title("AUC of each number column on its own (pink = my top 4)")
ax.set_xlabel("AUC alone  (0.5 = no signal)")
plt.tight_layout(); plt.show()
TOP_NUM = list(solo.sort_values(ascending=False).index[:6])
print("strongest six:", TOP_NUM)
'''),
    md('''
## Risk curves: how does the fraud rate change along each strong column?

I split each column into groups (each with about the same number of transactions; small columns like
`cvv_retry_count` keep their own values) and plotted the fraud rate of each group. The shaded band is the
95% interval.
'''),
    code(r'''
def risk_curve(ax, col, bins=20):
    s = train[col]
    if s.nunique() <= 25:
        key = s
    else:
        key = pd.qcut(s, q=bins, duplicates="drop")
    g = train.groupby(key, observed=True)[TARGET].agg(frauds="sum", n="size")
    x = s.groupby(key, observed=True).median().values
    rate = g.frauds / g.n
    lo, hi = wilson(g.frauds, g.n)
    ax.fill_between(x, lo, hi, color=PINK, alpha=0.18, lw=0)
    ax.plot(x, rate, color=PINK, marker="o", ms=4)
    ax.axhline(y.mean(), color=PURPLE, ls="--", lw=1.2)
    ax.set_title(col, fontsize=11)
    if s.nunique() <= 25:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.ravel(), TOP_NUM):
    risk_curve(ax, col)
fig.suptitle("Fraud rate along my six strongest columns (dashed = overall rate)", fontweight="bold", color="#4a1f73")
plt.tight_layout(); plt.show()
'''),
    note("A curve that goes up AND down 🎢",
         "Not every curve is a straight line. When the fraud rate rises and then falls again (look at the "
         "risk-score curve), a plain logistic regression can't follow it, because it can only draw "
         "\"more = riskier\" or \"more = safer\". Decision trees can follow any shape. That's one reason trees "
         "do well here."),
    md('''
## Two things together: authentication × wrong CVV attempts

Joe's favourite fraud signal is when someone types the CVV wrong several times. I wanted to see what
happens when that combines with a weak login method. Each square shows the fraud rate (and how many
transactions are in it).
'''),
    code(r'''
pv = train.pivot_table(index="auth_method", columns="cvv_retry_count", values=TARGET, aggfunc="mean")
nv = train.pivot_table(index="auth_method", columns="cvv_retry_count", values=TARGET, aggfunc="size")
order = pv.mean(axis=1).sort_values().index
pv, nv = pv.loc[order], nv.loc[order]
fig, ax = plt.subplots(figsize=(10, 4.8))
im = ax.imshow(pv.values, cmap=SEQ, aspect="auto")
ax.set_xticks(range(pv.shape[1]), pv.columns); ax.set_yticks(range(pv.shape[0]), pv.index)
ax.set_xlabel("wrong CVV attempts"); ax.grid(False)
for i in range(pv.shape[0]):
    for j in range(pv.shape[1]):
        v, n = pv.values[i, j], nv.values[i, j]
        if np.isnan(v):
            continue
        dark = v > np.nanmax(pv.values) * 0.55
        ax.text(j, i, f"{v:.0%}\n(n={int(n):,})", ha="center", va="center", fontsize=8.5,
                color="white" if dark else INK)
fig.colorbar(im, ax=ax, format=plt.FuncFormatter(lambda v, _: f"{v:.0%}"), label="fraud rate")
ax.set_title("Fraud rate: authentication method × wrong CVV attempts")
plt.tight_layout(); plt.show()
i, j = np.unravel_index(np.nanargmax(pv.values), pv.shape)
print(f"riskiest square: {pv.index[i]} with {pv.columns[j]} wrong CVV attempts -> {pv.values[i, j]:.1%} fraud "
      f"({int(nv.values[i, j]):,} transactions)")
'''),
    code(r'''
# How many of the six yes/no warning flags are switched on?
flag_count = train[FLAGS].sum(axis=1)
t = train.assign(flags_on=flag_count).groupby("flags_on")[TARGET].agg(frauds="sum", n="size")
t["rate"] = t.frauds / t.n
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(t.index, t.rate, color=PINK, width=0.6)
for x_, r, n in zip(t.index, t.rate, t.n):
    ax.text(x_, r + t.rate.max() * 0.02, f"{r:.1%}\nn={n:,}", ha="center", va="bottom", fontsize=8.5, color=SOFT_INK)
ax.axhline(y.mean(), color=PURPLE, ls="--", lw=1.5)
ax.set_xlabel("number of warning flags switched on"); ax.set_title("More warning flags → more fraud")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_ylim(0, t.rate.max() * 1.25)
plt.tight_layout(); plt.show()
no_flag_share = train.loc[flag_count == 0, TARGET].sum() / train[TARGET].sum()
print(f"...but {no_flag_share:.0%} of all frauds have NO warning flag at all -> flags alone can't catch everything")
'''),

    # ---------------------------------------------------------------- 4
    section(4, "A real Simpson's paradox", "A pattern that flips when you look inside groups"),
    md('''
I read about Simpson's paradox in another EDA notebook (*"Fraud Beneath the Surface"* — thank you!) and
wanted to check it myself.

**Question:** are transactions that spend a big part of the account balance (more than 10%) riskier?

First I compare the two groups **overall**. Then I compare them again **inside each amount group**
(cheap purchases with cheap purchases, expensive with expensive). If the answer flips, that's Simpson's
paradox: the overall comparison was really about *amount*, not about the balance ratio.
'''),
    code(r'''
ratio = train["amount_usd"] / train["account_balance_usd"].clip(lower=1)
big = ratio > 0.10
overall_big, overall_small = train.loc[big, TARGET].mean(), train.loc[~big, TARGET].mean()
decile = pd.qcut(train["amount_usd"], 10, labels=False, duplicates="drop")
inside = train.groupby([decile, big])[TARGET].mean().unstack()
inside.columns = ["spends ≤10% of balance", "spends >10% of balance"]

fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 2.2]})
axes[0].bar(["≤10%", ">10%"], [overall_small, overall_big], color=[PURPLE, PINK], width=0.55)
for x_, v in enumerate([overall_small, overall_big]):
    axes[0].text(x_, v, f"{v:.2%}", ha="center", va="bottom", color=SOFT_INK)
axes[0].set_title("Overall"); axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
for col, colr, mk in [("spends ≤10% of balance", PURPLE, "o"), ("spends >10% of balance", PINK, "s")]:
    axes[1].plot(inside.index + 1, inside[col], color=colr, marker=mk, label=col)
    axes[1].text(inside.index[-1] + 1.15, inside[col].iloc[-1], col.split(" ")[1], color=colr, va="center", fontsize=9)
axes[1].set_xlabel("amount group (1 = cheapest 10%, 10 = most expensive 10%)")
axes[1].set_title("Inside each amount group"); axes[1].legend(loc="upper left")
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
plt.tight_layout(); plt.show()

big_riskier_overall = overall_big > overall_small
wins_inside = int((inside.iloc[:, 1] > inside.iloc[:, 0]).sum())
print(f"overall: >10% group {overall_big:.2%} vs ≤10% group {overall_small:.2%} "
      f"-> big-ratio looks {'RISKIER' if big_riskier_overall else 'SAFER'} overall")
print(f"inside the {len(inside)} amount groups, the >10% group is riskier in {wins_inside} of them")
flipped = (big_riskier_overall and wins_inside < len(inside) / 2) or (not big_riskier_overall and wins_inside > len(inside) / 2)
print("=> Simpson's paradox: the direction FLIPS inside groups! 🤯" if flipped else
      "=> no flip here: the inside-group pattern agrees with the overall one")
'''),
    note("Why this matters to me 💡",
         "If I had only looked at the overall bars, I might have built a feature on a pattern that isn't really "
         "there. Checking inside groups is like asking \"is this still true when I compare fair pairs?\""),

    # ---------------------------------------------------------------- 5
    section(5, "Clues left by the data generator", "The digits of a number carry signal here (but wouldn't in real life)"),
    md('''
This data is **synthetic** — a computer generated it from a real dataset. Other Kagglers discovered that the
generator writes numbers *digit by digit*, so the **two digits after the decimal point** ("cents") can carry
fraud signal on their own. That sounded too weird to be true, so I checked.

Each bar below is one possible ending (`.00` to `.99`). Bars much higher than the dashed line are endings
where fraud is unusually common.
'''),
    code(r'''
def cents_chart(ax, col):
    cents = (np.rint(train[col] * 100).astype(int) % 100)
    g = train.groupby(cents)[TARGET].agg(frauds="sum", n="size")
    rate = g.frauds / g.n
    hot = rate > y.mean() * 1.5
    ax.bar(rate.index, rate, color=np.where(hot, PINK, LAVENDER), width=0.8)
    ax.axhline(y.mean(), color=PURPLE, ls="--", lw=1.5)
    ax.set_title(f"Fraud rate by the 2 digits after the decimal point: {col}", fontsize=11)
    ax.set_xlabel("ending (cents)"); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    top = rate.sort_values(ascending=False).head(3)
    return ", ".join(f".{k:02d} → {v:.1%} (n={g.n[k]:,})" for k, v in top.items())

fig, axes = plt.subplots(2, 1, figsize=(15, 7.5))
for ax, col in zip(axes, ["hours_since_last_txn", "amount_usd"]):
    print(f"{col}: riskiest endings  {cents_chart(ax, col)}")
plt.tight_layout(); plt.show()
'''),
    note("Real life vs. this competition 🌍",
         "In a real bank, the cents of a purchase shouldn't tell you anything about fraud. Here they do, "
         "because of how the data was made. I use this trick to score well, but I wrote it down as an "
         "<b>artifact</b> of the competition, not a real fraud pattern."),

    # ---------------------------------------------------------------- 6
    section(6, "Does PCA find fraud?", "Most variance ≠ most fraud signal"),
    md('''
I learned PCA (principal component analysis) before this competition, so my first idea for choosing
features was: "use PCA and keep the top components!"

But PCA ranks directions by how much the numbers **vary** — it never looks at `is_fraud`. So I measured both
things for every component: how much variance it explains, and how well it separates fraud (AUC).
'''),
    code(r'''
Z = StandardScaler().fit_transform(train[NUM].astype(float))
pca = PCA().fit(Z)
pcs = pca.transform(Z)
pc_auc = np.array([max(roc_auc_score(y, pcs[:, i]), 1 - roc_auc_score(y, pcs[:, i])) for i in range(pcs.shape[1])])
names = [f"PC{i + 1}" for i in range(len(pc_auc))]

fig, axes = plt.subplots(1, 2, figsize=(15, 4.3), sharex=True)
axes[0].bar(names, pca.explained_variance_ratio_, color=PURPLE, width=0.6)
axes[0].set_title("How much each component VARIES"); axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
axes[1].bar(names, pc_auc - 0.5, bottom=0.5, color=PINK, width=0.6)
axes[1].axhline(0.5, color=PURPLE, ls="--", lw=1.2)
axes[1].set_title("How well each component finds FRAUD (AUC)")
for ax in axes:
    ax.tick_params(axis="x", rotation=60)
plt.tight_layout(); plt.show()
best = int(np.argmax(pc_auc))
print(f"PC1 explains the most variance ({pca.explained_variance_ratio_[0]:.1%}) and has fraud AUC {pc_auc[0]:.3f}")
print(f"the best component for fraud is {names[best]} (AUC {pc_auc[best]:.3f}), "
      f"which explains only {pca.explained_variance_ratio_[best]:.1%} of the variance")
'''),
    note("My takeaway 🎓",
         "PCA is great for squeezing data, but it doesn't know what I'm trying to predict. To decide which "
         "features to keep, I instead <b>remove a group of features and re-check my score</b> (section 10)."),

    # ---------------------------------------------------------------- 7
    section(7, "Target encoding — and the leakage trap", "The most useful trick here, and the easiest one to get wrong"),
    md('''
**Target encoding** replaces each value with the **fraud rate of rows that have that value**. For rare values
the rate is unreliable, so it gets pulled toward the overall fraud rate. I use the simple *m-estimate*
(m = 20), the same as the strongest public recipe:

$$\\text{encoding} = \\frac{\\text{frauds with this value} + 20 \\times \\text{overall rate}}{\\text{rows with this value} + 20}$$
'''),
    note("By hand ✏️",
         "Overall fraud rate 2%. A value seen <b>5</b> times with <b>1</b> fraud: (1 + 20×0.02) / (5 + 20) = "
         "1.4 / 25 = <b>0.056</b> — the raw rate 0.20 gets pulled way down, because 5 rows isn't much evidence.<br>"
         "A value seen <b>500</b> times with <b>100</b> frauds: (100 + 0.4) / 520 = <b>0.193</b> — barely pulled, "
         "because 500 rows is a lot of evidence."),
    md('''
## ⚠️ The trap: leakage

If a row's **own label** is used to compute its encoding, the encoding secretly contains the answer. To prove
it to myself, I made a column of **pure random noise** — it has nothing to do with fraud — and encoded it
two ways:

1. **Wrong:** compute each value's fraud rate using *all* rows (including the row itself)
2. **Right:** split the rows into 5 parts; each part is encoded using only the *other* 4 parts ("cross-fitting")

Then I measured the AUC of the encoded noise.
'''),
    code(r'''
def m_rates(keys, labels, prior, m=20.0):
    s = pd.DataFrame({"k": keys, "y": labels}).groupby("k", sort=False)["y"].agg(["sum", "count"])
    return (s["sum"] + m * prior) / (s["count"] + m)

def te_fit_transform(train_keys, labels, other_frames=(), m=20.0, inner=5, seed=0):
    """Cross-fitted m-estimate encoding of every column in train_keys.
    Train rows: encoded by the other inner folds (never their own label).
    Other frames (validation / test): encoded with rates from ALL training rows."""
    labels = np.asarray(labels, float)
    prior = labels.mean()
    out_tr = pd.DataFrame(index=train_keys.index)
    outs = [pd.DataFrame(index=o.index) for o in other_frames]
    folds = list(StratifiedKFold(inner, shuffle=True, random_state=seed).split(np.zeros(len(labels)), labels))
    for c in train_keys.columns:
        k = train_keys[c].to_numpy()
        enc = np.empty(len(k), dtype=np.float32)
        for a, b in folds:
            enc[b] = pd.Series(k[b]).map(m_rates(k[a], labels[a], prior, m)).fillna(prior).to_numpy()
        out_tr[f"te_{c}"] = enc
        full = m_rates(k, labels, prior, m)
        for o, frame in zip(outs, other_frames):
            o[f"te_{c}"] = pd.Series(frame[c].to_numpy()).map(full).fillna(prior).to_numpy().astype(np.float32)
    return out_tr, outs

rng = np.random.default_rng(7)
noise = pd.DataFrame({"noise": rng.integers(0, 30_000, len(train))})      # random numbers, no relation to fraud
wrong = noise["noise"].map(m_rates(noise["noise"].to_numpy(), y, y.mean())).to_numpy()   # uses each row's own label
right, _ = te_fit_transform(noise, y)                                      # cross-fitted
auc_wrong, auc_right = roc_auc_score(y, wrong), roc_auc_score(y, right["te_noise"])

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.barh(["right way\n(cross-fitted)", "wrong way\n(uses own label)"], [auc_right - 0.5, auc_wrong - 0.5],
        left=0.5, color=[PURPLE, PINK], height=0.5)
for i, v in enumerate([auc_right, auc_wrong]):
    ax.text(v + 0.005, i, f"AUC {v:.3f}", va="center", color=SOFT_INK)
ax.axvline(0.5, color=INK, lw=1)
ax.set_xlim(0.45, max(auc_wrong, 0.6) + 0.08)
ax.set_title("The SAME random-noise column, encoded two ways")
plt.tight_layout(); plt.show()
print(f"wrong way: AUC {auc_wrong:.3f}  <- looks like a great feature, but it's pure noise!")
print(f"right way: AUC {auc_right:.3f}  <- honest: noise is worth nothing")
'''),
    note("This one scared me 😱",
         "The wrong way makes random noise look like a strong fraud signal. On the leaderboard it would fall "
         "apart, because test rows don't have labels to peek at. From now on, every encoding in this notebook is "
         "<b>cross-fitted and computed inside each CV fold</b>."),
    md('''
## The digit keys I encode

Because of the generator clue (section 5), I don't just encode each number — I also encode **how it is
written**. Example for `1371.49`:

| key | value | meaning |
|---|---|---|
| cents | 49 | the two digits after the decimal point (also kept as a normal column) |
| int | 1371 | whole part (cut off, not rounded) |
| ten | 137 | whole part without its last digit |
| dec | 13714 | whole part + first decimal digit |
| l1 | 41 | **4** digits long, starts with **1** |
| l2 | 413 | **4** digits long, starts with **13** |

The key columns are only used to make encodings; afterwards I drop them so the trees focus on the useful
columns. (Recipe credit: *goodpjw2008*, shared in *alexchilton*'s "Shallow-Tree Stack" notebook.)
'''),
    code(r'''
def add_digit_keys(df, cols=None):
    df = df.copy()
    for c in (FLOATS if cols is None else cols):
        x = df[c].astype(float).to_numpy()
        whole = np.floor(x).astype("int64")
        s = pd.Series(whole).astype(str)
        n = s.str.len().to_numpy()
        df[f"cents_{c}"] = (np.rint(x * 100).astype("int64") % 100).astype("int16")
        df[f"key_int_{c}"] = whole
        df[f"key_ten_{c}"] = np.floor(x / 10).astype("int64")
        df[f"key_dec_{c}"] = np.floor(x * 10 + 1e-6).astype("int64")
        df[f"key_l1_{c}"] = n * 10 + s.str[0].astype("int64").to_numpy()
        df[f"key_l2_{c}"] = n * 100 + s.str[:2].astype("int64").to_numpy()
    return df

print("check with my example 1371.49:")
display(add_digit_keys(pd.DataFrame({"amount_usd": [1371.49]}), cols=["amount_usd"]).T.rename(columns={0: "value"}))

trk, tek = add_digit_keys(train), add_digit_keys(test)
CENTS = [c for c in trk.columns if c.startswith("cents_")]
KEYS = [c for c in trk.columns if c.startswith("key_")]
ENCODE = FEATS + CENTS + KEYS                                   # every column gets a fraud-rate encoding
for frame in (trk, tek):
    for c in CAT:
        frame[c] = frame[c].astype("category")
    for c in FLAGS:
        frame[c] = frame[c].astype(int)
cats_all = {c: pd.api.types.union_categoricals([trk[c], tek[c]]).categories for c in CAT}
for frame in (trk, tek):
    for c in CAT:
        frame[c] = frame[c].cat.set_categories(cats_all[c])
print(f"{len(CENTS)} cents columns, {len(KEYS)} helper keys, {len(ENCODE)} columns to encode")
'''),

    # ---------------------------------------------------------------- 8
    section(8, "Why small trees win", "I measured trees with 2, 3, 4, 8, 16 and 63 leaves"),
    md('''
My first LightGBM used the default **63 leaves** per tree. A public notebook said **tiny trees** (3–4 leaves)
do better here, and I wanted to see it with my own eyes. I used one fold (80% train / 20% check) and
changed only `num_leaves`.

**My guess before running:** if the fraud signal is mostly "add up a risk for each column" (like logistic
regression does), big trees spend their splits on combinations that are mostly noise.
'''),
    code(r'''
import lightgbm as lgb

def make_frames(tr_idx, va_idx=None, seed=0, with_keys=False):
    """Encodings fitted on the training rows only; validation and test rows use those rates.
    va_idx=None -> no validation part (used for the final fit on all rows).
    with_keys=True keeps the helper digit keys (the joint GLM one-hots them)."""
    a = trk.iloc[tr_idx]
    others = [tek[ENCODE].reset_index(drop=True)]
    if va_idx is not None:
        others.insert(0, trk.iloc[va_idx][ENCODE].reset_index(drop=True))
    te_a, te_o = te_fit_transform(a[ENCODE].reset_index(drop=True), y[tr_idx], others, seed=seed)
    keep = FEATS + CENTS + (KEYS if with_keys else [])
    Xa = pd.concat([a[keep].reset_index(drop=True), te_a], axis=1)
    Xt = pd.concat([tek[keep].reset_index(drop=True), te_o[-1]], axis=1)
    Xb = None
    if va_idx is not None:
        Xb = pd.concat([trk.iloc[va_idx][keep].reset_index(drop=True), te_o[0]], axis=1)
    return Xa, Xb, Xt

FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=42).split(trk, y))
tr0, va0 = FOLDS[0]
Xa, Xb, _ = make_frames(tr0, va0)
leaf_rows = []
for leaves in [2, 3, 4, 8, 16, 63]:
    t0 = time.time()
    m = lgb.LGBMClassifier(n_estimators=8000, learning_rate=0.1, num_leaves=leaves, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1, random_state=0)
    m.fit(Xa, y[tr0], eval_set=[(Xb, y[va0])], eval_metric="auc", callbacks=[lgb.early_stopping(150, verbose=False)])
    auc = roc_auc_score(y[va0], m.predict_proba(Xb)[:, 1])
    leaf_rows.append({"leaves": leaves, "AUC": auc, "trees used": m.best_iteration_, "seconds": time.time() - t0})
    print(f"{leaves:>3} leaves: AUC {auc:.5f}   ({m.best_iteration_} trees, {time.time() - t0:.0f}s)")
leaf_df = pd.DataFrame(leaf_rows)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(leaf_df.leaves, leaf_df.AUC, color=PINK, marker="o", ms=8)
best = leaf_df.loc[leaf_df.AUC.idxmax()]
ax.annotate(f"best: {int(best.leaves)} leaves", (best.leaves, best.AUC), textcoords="offset points",
            xytext=(10, -18), color=PURPLE, fontweight="bold")
ax.set_xscale("log", base=2); ax.set_xticks(leaf_df.leaves, leaf_df.leaves)
ax.set_xlabel("leaves per tree"); ax.set_ylabel("validation AUC")
ax.set_title("Smaller trees, better score (one fold)")
plt.tight_layout(); plt.show()
'''),

    # ---------------------------------------------------------------- 9
    section(9, "My first three models: Logistic, LightGBM, CatBoost", "5-fold cross-validation, then a first blend"),
    md('''
Now the real thing. I use **5-fold stratified cross-validation**: split the training data into 5 parts,
train on 4, predict the 5th, and rotate. Every row gets exactly one honest prediction ("out-of-fold", OOF),
and the AUC of those is my **CV score**. "Stratified" keeps the fraud rate the same in every part.

Three different kinds of model (different models make different mistakes, which helps a blend):

| model | what it is | settings |
|---|---|---|
| 💜 **Logistic regression** | a straight-line model, on the log-odds of every encoding | `C=0.05` |
| 🌸 **LightGBM** | many tiny trees, each fixing the mistakes of the ones before | 3 leaves, early stopping |
| 🐱 **CatBoost** | boosting with "symmetric" trees and its own way of handling categories | depth 4, early stopping |

All encodings are made **inside each fold** from the training part only.
'''),
    code(r'''
from catboost import CatBoostClassifier

def logodds(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))

oof = {k: np.zeros(len(trk)) for k in ["logistic", "lightgbm", "catboost"]}
pred = {k: np.zeros(len(tek)) for k in oof}
ITS = {"lightgbm": [], "catboost": []}     # trees used in each fold (for the full refit later)
FRAMES = {}                                # each fold's encoded data, kept for the later models
lgb_imp = None
for k, (tr_i, va_i) in enumerate(FOLDS):
    t0 = time.time()
    FRAMES[k] = make_frames(tr_i, va_i, seed=k, with_keys=True)
    Xa, Xb, Xt = (d.drop(columns=KEYS) for d in FRAMES[k])
    te_cols = [c for c in Xa.columns if c.startswith("te_")]

    # 💜 logistic regression on the encodings (as log-odds, then scaled)
    sc = StandardScaler().fit(logodds(Xa[te_cols]))
    lr = LogisticRegression(C=0.05, max_iter=2000).fit(sc.transform(logodds(Xa[te_cols])), y[tr_i])
    oof["logistic"][va_i] = lr.predict_proba(sc.transform(logodds(Xb[te_cols])))[:, 1]
    pred["logistic"] += lr.predict_proba(sc.transform(logodds(Xt[te_cols])))[:, 1] / len(FOLDS)

    # 🌸 LightGBM, 3 leaves
    m = lgb.LGBMClassifier(n_estimators=20000, learning_rate=0.05, num_leaves=3, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                           verbose=-1, random_state=k)
    m.fit(Xa, y[tr_i], eval_set=[(Xb, y[va_i])], eval_metric="auc", callbacks=[lgb.early_stopping(300, verbose=False)])
    oof["lightgbm"][va_i] = m.predict_proba(Xb)[:, 1]
    pred["lightgbm"] += m.predict_proba(Xt)[:, 1] / len(FOLDS)
    ITS["lightgbm"].append(m.best_iteration_ or 1000)
    imp = pd.Series(m.booster_.feature_importance("gain") if hasattr(m, "booster_") else m.feature_importances_, index=Xa.columns)
    lgb_imp = imp if lgb_imp is None else lgb_imp + imp

    # 🐱 CatBoost (it reads the word columns itself)
    to_str = lambda d: d.assign(**{c: d[c].astype(str) for c in CAT})
    cb = CatBoostClassifier(iterations=4000, learning_rate=0.1, depth=4, eval_metric="AUC", random_seed=k,
                            od_type="Iter", od_wait=200, verbose=0, thread_count=-1)
    cb.fit(to_str(Xa), y[tr_i], cat_features=CAT, eval_set=(to_str(Xb), y[va_i]))
    oof["catboost"][va_i] = cb.predict_proba(to_str(Xb))[:, 1]
    pred["catboost"] += cb.predict_proba(to_str(Xt))[:, 1] / len(FOLDS)
    ITS["catboost"].append(cb.get_best_iteration() + 1 if hasattr(cb, "get_best_iteration") else 1000)

    print(f"fold {k + 1}: " + "  ".join(f"{n} {roc_auc_score(y[va_i], oof[n][va_i]):.5f}" for n in oof)
          + f"   ({time.time() - t0:.0f}s)")

cv = {n: roc_auc_score(y, o) for n, o in oof.items()}
print("\nCV AUC:", {n: round(v, 5) for n, v in cv.items()})
'''),
    md('''
## Blending: are my models different enough to help each other?

AUC only cares about order, so before blending I turn every model's predictions into **ranks**, then try
every mix of weights (steps of 0.1) and keep the best one. I also check how similar the models are: if two
models rank transactions almost identically (rank correlation near 1.0), blending them adds almost nothing.
'''),
    code(r'''
import itertools
names = list(oof)
R = np.column_stack([rankdata(oof[n]) / len(trk) for n in names])
best_w, best_auc = None, -1
for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=len(names)):
    if abs(sum(w) - 1) > 1e-9:
        continue
    s = roc_auc_score(y, R @ np.array(w))
    if s > best_auc:
        best_w, best_auc = np.array(w), s
cv["blend"] = best_auc
corr = pd.DataFrame(np.corrcoef(R.T), index=names, columns=names)

fig, axes = plt.subplots(1, 2, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1.5, 1]})
lab = list(cv)
cols_ = [PURPLE, PINK, TEAL, "#4a1f73"]
axes[0].barh(lab, [cv[n] for n in lab], color=cols_, height=0.55)
for i, n in enumerate(lab):
    axes[0].text(cv[n] + 0.0002, i, f"{cv[n]:.5f}", va="center", color=SOFT_INK)
lo_ = min(cv.values()) - 0.002
axes[0].set_xlim(lo_, max(cv.values()) + 0.0025); axes[0].set_title("CV AUC of each model and the blend")
vmin_ = min(0.8, corr.values.min())
im = axes[1].imshow(corr.values, cmap=SEQ, vmin=vmin_, vmax=1)
axes[1].set_xticks(range(len(names)), names); axes[1].set_yticks(range(len(names)), names); axes[1].grid(False)
for i in range(len(names)):
    for j in range(len(names)):
        axes[1].text(j, i, f"{corr.values[i, j]:.3f}", ha="center", va="center",
                     color="white" if corr.values[i, j] > vmin_ + 0.6 * (1 - vmin_) else INK, fontsize=9)
axes[1].set_title("How alike are the models? (rank correlation)")
plt.tight_layout(); plt.show()
print("best blend weights:", dict(zip(names, best_w.round(1))), f"-> blend CV AUC {best_auc:.5f}")
'''),

    # ---------------------------------------------------------------- 10
    section(10, "A smarter straight-line model: the joint GLM", "Logistic regression with much better inputs"),
    md('''
My logistic regression in section 9 only saw the encodings. A public notebook (credit: *goodpjw2008* and
*lazyyeou*) showed a much stronger version, called a **joint GLM** ("generalized linear model"). It is
still logistic regression, but its inputs are:

| input | why |
|---|---|
| log-odds of every target encoding | the encodings, on the scale logistic regression likes |
| every number, scaled | the plain values |
| **splines** of every number | smooth "bumps" along each number, so a straight-line model can still follow a curve that goes up *and* down |
| a **one-hot column for every exact value and every digit key seen at least 5 times** | lets the model correct each encoding for all the others at once; a rare value falls back on its coarser digit keys |

That is tens of thousands of columns, so **regularization** matters a lot. On my laptop I tried six
strengths (`C` is the *inverse* of the strength — smaller `C` holds the model back more):

| C | 0.003 | **0.01** | 0.03 | 0.1 | 0.3 | 1.0 |
|---|---|---|---|---|---|---|
| CV AUC | 0.94838 | **0.94936** | 0.94929 | 0.94798 | 0.94484 | 0.93792 |

A perfect overfitting curve: hold back too much and it misses signal; too little and it memorizes noise.
I use `C = 0.01`.
'''),
    code(r'''
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))

class JointGLM:
    """Logistic regression on: log-odds of every encoding, scaled numbers, splines of every number,
    and a one-hot column for every exact value / digit key seen at least 5 times."""
    def __init__(self, C=0.01):
        self.C = C
    def _keys(self, X):
        return X[ENCODE].astype({c: str for c in CAT})
    def _dense(self, X):
        te = [c for c in X.columns if c.startswith("te_")]
        return np.column_stack([logodds(X[te]), X[NUM + CENTS].to_numpy(float)])
    def _design(self, X):
        return sparse.hstack([sparse.csr_matrix(self.sc_.transform(self._dense(X))),
                              sparse.csr_matrix(self.spl_.transform(X[NUM].to_numpy(float))),
                              self.oh_.transform(self._keys(X))]).tocsr()
    def fit(self, X, labels):
        self.sc_ = StandardScaler().fit(self._dense(X))
        self.spl_ = SplineTransformer(n_knots=8, degree=3, knots="uniform", extrapolation="constant").fit(X[NUM].to_numpy(float))
        self.oh_ = OneHotEncoder(min_frequency=5, handle_unknown="infrequent_if_exist").fit(self._keys(X))
        self.lr_ = LogisticRegression(C=self.C, max_iter=1000).fit(self._design(X), labels)
        return self
    def logit(self, X):
        return self.lr_.decision_function(self._design(X))

oof["joint_glm"], pred["joint_glm"] = np.zeros(len(trk)), np.zeros(len(tek))
for k, (tr_i, va_i) in enumerate(FOLDS):
    t0 = time.time()
    Xa, Xb, Xt = FRAMES[k]
    g = JointGLM().fit(Xa, y[tr_i])
    oof["joint_glm"][va_i] = sigmoid(g.logit(Xb))
    pred["joint_glm"] += sigmoid(g.logit(Xt)) / len(FOLDS)
    print(f"fold {k + 1}: joint GLM {roc_auc_score(y[va_i], oof['joint_glm'][va_i]):.5f}   ({time.time() - t0:.0f}s)")
cv["joint_glm"] = roc_auc_score(y, oof["joint_glm"])
print(f"\njoint GLM CV AUC {cv['joint_glm']:.5f}   (my plain logistic in section 9: {cv['logistic']:.5f})")
'''),

    # ---------------------------------------------------------------- 11
    section(11, "Trees that start where the GLM stopped", "GLM-boost: the trees only learn the GLM's mistakes"),
    md('''
Here's an idea I really like. Normally boosting starts from nothing and the trees learn everything. In
**GLM-boost**, the trees start from the joint GLM's answer (its log-odds) and only learn the **correction**
— what the straight-line model gets wrong. LightGBM does this with `init_score`.

One trap (the same leakage idea as section 7!): the GLM answers used as the starting point for the
**training** rows must be *out-of-fold*, from an inner 3-way split. Otherwise the trees would start from
answers that already "saw" those labels.
'''),
    note("By hand ✏️",
         "Say the GLM gives a transaction log-odds of <b>−2.0</b> (about 12% fraud). The trees look at it and add "
         "<b>+0.5</b> because, say, it's gift cards at night with a weird ending in the cents. Final log-odds "
         "−2.0 + 0.5 = <b>−1.5</b> → about 18% fraud. The trees never have to re-learn what the GLM already knew."),
    code(r'''
GB_PARAMS = dict(learning_rate=0.03, num_leaves=4, min_child_samples=100, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.5, reg_lambda=1.0, verbose=-1)

def glm_boost_fit(Xa, ya, Xb=None, yb=None, n_trees=None, seed=0):
    """Returns (glm, trees). Training rows start from inner out-of-fold GLM log-odds."""
    start = np.zeros(len(Xa))
    for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(Xa, ya):
        start[b] = JointGLM().fit(Xa.iloc[a], ya[a]).logit(Xa.iloc[b])
    g = JointGLM().fit(Xa, ya)
    m = lgb.LGBMClassifier(n_estimators=n_trees or 20000, random_state=seed, **GB_PARAMS)
    if Xb is not None:
        m.fit(Xa.drop(columns=KEYS), ya, init_score=start, eval_set=[(Xb.drop(columns=KEYS), yb)],
              eval_init_score=[g.logit(Xb)], eval_metric="auc", callbacks=[lgb.early_stopping(300, verbose=False)])
    else:
        m.fit(Xa.drop(columns=KEYS), ya, init_score=start)
    return g, m

def glm_boost_logit(g, m, X):
    return g.logit(X) + m.predict(X.drop(columns=KEYS), raw_score=True)   # GLM answer + tree correction

oof["glm_boost"], pred["glm_boost"] = np.zeros(len(trk)), np.zeros(len(tek))
ITS["glm_boost"] = []
for k, (tr_i, va_i) in enumerate(FOLDS):
    t0 = time.time()
    Xa, Xb, Xt = FRAMES[k]
    g, m = glm_boost_fit(Xa, y[tr_i], Xb, y[va_i], seed=k)
    ITS["glm_boost"].append(m.best_iteration_ or 1000)
    oof["glm_boost"][va_i] = sigmoid(glm_boost_logit(g, m, Xb))
    pred["glm_boost"] += sigmoid(glm_boost_logit(g, m, Xt)) / len(FOLDS)
    print(f"fold {k + 1}: GLM-boost {roc_auc_score(y[va_i], oof['glm_boost'][va_i]):.5f}   "
          f"({m.best_iteration_} trees on top of the GLM, {time.time() - t0:.0f}s)")
cv["glm_boost"] = roc_auc_score(y, oof["glm_boost"])
print(f"\nGLM-boost CV AUC {cv['glm_boost']:.5f}   (joint GLM alone {cv['joint_glm']:.5f}, LightGBM alone {cv['lightgbm']:.5f})")
'''),

    # ---------------------------------------------------------------- 12
    section(12, "A neural network", "Smooth curves instead of steps — does it help?"),
    md('''
Trees cut the data into boxes, so they draw **steps**. A neural network draws **smooth curves**, so it
should make *different* mistakes — which is exactly what a blend wants. Some top public notebooks got their
later gains from networks.

My network is a small **multi-layer perceptron**: inputs → 256 → 128 → 1, with SiLU activations and
dropout. Before training, every number (including the encodings) goes through a **quantile transform**,
which reshapes each column into a bell curve so no column dominates; word columns are one-hot. I train
with AdamW, check validation AUC after every pass over the data ("epoch"), and keep the best epoch — early
stopping, just like the trees.

**Honest spoiler:** on my laptop the network scored a bit below the trees and got almost no weight in the
blend. I'm keeping it because a negative result is still a result. Let's see what happens here.
'''),
    code(r'''
from sklearn.preprocessing import QuantileTransformer
try:
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
except ImportError:                       # (Kaggle has PyTorch; this is only for odd environments)
    torch, DEV = None, "none"
print("neural network device:", DEV)
EPOCHS = 6 if FAST else 30

def nn_matrices(Xa, others):
    nums = [c for c in Xa.columns if c.startswith("te_")] + NUM + CENTS + FLAGS
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200_000,
                             random_state=0).fit(Xa[nums].to_numpy(np.float32))
    def mat(X):
        parts = [qt.transform(X[nums].to_numpy(np.float32)).astype(np.float32)]
        parts += [pd.get_dummies(X[c]).to_numpy(np.float32) for c in CAT]   # categories are fixed, so columns line up
        return np.concatenate(parts, axis=1)
    return mat(Xa), [mat(o) for o in others]

def train_mlp(A, ya, V, yv, seed=0):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(A.shape[1], 256), torch.nn.SiLU(), torch.nn.Dropout(0.15),
                              torch.nn.Linear(256, 128), torch.nn.SiLU(), torch.nn.Dropout(0.15),
                              torch.nn.Linear(128, 1)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    At, yt = torch.from_numpy(A), torch.from_numpy(ya.astype(np.float32))
    best, best_state, bad = -1, None, 0
    for ep in range(EPOCHS):
        net.train()
        for idx in torch.randperm(len(At)).split(4096):
            opt.zero_grad()
            loss_fn(net(At[idx].to(DEV)).squeeze(1), yt[idx].to(DEV)).backward()
            opt.step()
        auc = roc_auc_score(yv, mlp_predict(net, V))
        if auc > best:
            best, bad, best_state = auc, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 4:
                break
    net.load_state_dict(best_state)
    return net

def mlp_predict(net, M):
    net.eval()
    with torch.no_grad():
        return np.concatenate([sigmoid(net(torch.from_numpy(M[i:i + 65536]).to(DEV)).squeeze(1).float().cpu().numpy())
                               for i in range(0, len(M), 65536)])

oof["neural_net"], pred["neural_net"] = np.zeros(len(trk)), np.zeros(len(tek))
for k, (tr_i, va_i) in enumerate(FOLDS):
    t0 = time.time()
    Xa, Xb, Xt = FRAMES[k]
    A, (B, T) = nn_matrices(Xa, [Xb, Xt])
    if torch is not None:
        net = train_mlp(A, y[tr_i], B, y[va_i], seed=k)
        oof["neural_net"][va_i], p_t = mlp_predict(net, B), mlp_predict(net, T)
    else:
        from sklearn.neural_network import MLPClassifier
        mm = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=30, early_stopping=True, random_state=k).fit(A, y[tr_i])
        oof["neural_net"][va_i], p_t = mm.predict_proba(B)[:, 1], mm.predict_proba(T)[:, 1]
    pred["neural_net"] += p_t / len(FOLDS)
    print(f"fold {k + 1}: neural network {roc_auc_score(y[va_i], oof['neural_net'][va_i]):.5f}   ({time.time() - t0:.0f}s)")
cv["neural_net"] = roc_auc_score(y, oof["neural_net"])
print(f"\nneural network CV AUC {cv['neural_net']:.5f}")
'''),

    # ---------------------------------------------------------------- 13
    section(13, "Train once more on everything", "The full refit trick for the final predictions"),
    md('''
So far each model's **test** predictions are the average of 5 fold models, and each of those saw only 80%
of the training rows. For the final answer, I train each model **one more time on 100% of the rows**.
Boosted models get **1.25 ×** the average number of trees they used in the folds (25% more rows → a few
more trees).

This was my sneakiest lesson: on my laptop, the refit version had the **same CV score** as before but
scored **+0.0006 better on the leaderboard**. CV can't see this gain, because CV is measured *before* this
last step. (I keep the neural network's fold average, to save time.)
'''),
    code(r'''
t0 = time.time()
ALL = np.arange(len(trk))
XA, _, XT = make_frames(ALL, None, seed=0, with_keys=True)
XA_, XT_ = XA.drop(columns=KEYS), XT.drop(columns=KEYS)
te_all = [c for c in XA_.columns if c.startswith("te_")]
refit = {}

sc = StandardScaler().fit(logodds(XA_[te_all]))
lr = LogisticRegression(C=0.05, max_iter=2000).fit(sc.transform(logodds(XA_[te_all])), y)
refit["logistic"] = lr.predict_proba(sc.transform(logodds(XT_[te_all])))[:, 1]

n = int(round(np.mean(ITS["lightgbm"]) * 1.25))
m = lgb.LGBMClassifier(n_estimators=n, learning_rate=0.05, num_leaves=3, min_child_samples=100, subsample=0.8,
                       subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, verbose=-1, random_state=0).fit(XA_, y)
refit["lightgbm"] = m.predict_proba(XT_)[:, 1]
print(f"LightGBM refit: {n} trees")

n = int(round(np.mean(ITS["catboost"]) * 1.25))
cb = CatBoostClassifier(iterations=n, learning_rate=0.1, depth=4, random_seed=0, verbose=0, thread_count=-1)
cb.fit(to_str(XA_), y, cat_features=CAT)
refit["catboost"] = cb.predict_proba(to_str(XT_))[:, 1]
print(f"CatBoost refit: {n} trees")

refit["joint_glm"] = sigmoid(JointGLM().fit(XA, y).logit(XT))

n = int(round(np.mean(ITS["glm_boost"]) * 1.25))
g, m = glm_boost_fit(XA, y, n_trees=n)
refit["glm_boost"] = sigmoid(glm_boost_logit(g, m, XT))
print(f"GLM-boost refit: {n} trees on top of the GLM")

final_pred = {n_: refit.get(n_, pred[n_]) for n_ in oof}      # neural net keeps its fold average
print(f"full refit done in {(time.time() - t0) / 60:.1f} minutes")
'''),

    # ---------------------------------------------------------------- 14
    section(14, "The final blend: hill climbing", "Letting the data pick the mix of models"),
    md('''
With six models, trying every weight combination gets slow. Instead I use **hill climbing** (a classic
Kaggle method from Caruana et al., 2004):

1. Start with the single best model.
2. Try adding one more "copy" of each model to the mix, and keep the one that raises CV AUC the most.
3. Repeat until nothing helps.

A model can be added more than once, so the counts become the weights. Everything is done on **ranks**,
because AUC only cares about order.
'''),
    code(r'''
def hill_climb(oofs, labels, max_steps=60):
    names_ = list(oofs)
    R_ = {n_: rankdata(oofs[n_]) / len(labels) for n_ in names_}
    counts = dict.fromkeys(names_, 0)
    first = max(names_, key=lambda n_: roc_auc_score(labels, R_[n_]))
    counts[first], blend, score = 1, R_[first].copy(), roc_auc_score(labels, R_[first])
    path = [(first, score)]
    for _ in range(max_steps):
        total = sum(counts.values())
        tries = {n_: roc_auc_score(labels, (blend * total + R_[n_]) / (total + 1)) for n_ in names_}
        best_n = max(tries, key=tries.get)
        if tries[best_n] <= score + 1e-7:
            break
        blend, score = (blend * total + R_[best_n]) / (total + 1), tries[best_n]
        counts[best_n] += 1
        path.append((best_n, score))
    total = sum(counts.values())
    return {n_: c / total for n_, c in counts.items()}, score, path

weights, final_cv, path = hill_climb(oof, y)
cv["final blend"] = final_cv

fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
lab = sorted(oof, key=lambda n_: cv[n_]) + ["final blend"]
axes[0].barh(lab, [cv[n_] for n_ in lab], color=[LAVENDER] * (len(lab) - 1) + [PINK], height=0.55)
for i, n_ in enumerate(lab):
    axes[0].text(cv[n_] + 0.0002, i, f"{cv[n_]:.5f}", va="center", color=SOFT_INK)
axes[0].set_xlim(min(cv[n_] for n_ in lab) - 0.002, final_cv + 0.003)
axes[0].set_title("CV AUC of every model and the final blend")
w = pd.Series(weights).sort_values()
axes[1].barh(w.index, w.values, color=PURPLE, height=0.55)
for i, v in enumerate(w.values):
    axes[1].text(v + 0.01, i, f"{v:.0%}", va="center", color=SOFT_INK)
axes[1].set_xlim(0, max(w.values) + 0.15); axes[1].set_title("Weights picked by hill climbing")
axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
plt.tight_layout(); plt.show()
print("hill-climbing path:", " → ".join(f"{n_} ({s_:.5f})" for n_, s_ in path))
print(f"final blend CV AUC {final_cv:.5f}  vs best single model {max(cv[n_] for n_ in oof):.5f}")

# final submission: hill-climbing weights on the ranks of the refit test predictions
sub = pd.DataFrame({ID: test[ID], TARGET: sum(weights[n_] * rankdata(final_pred[n_]) / len(tek) for n_ in oof)})
sub.to_csv("submission.csv", index=False)
print(sub.shape, "-> submission.csv written")
'''),

    # ---------------------------------------------------------------- 15
    section(15, "Used a lot ≠ actually helpful", "Feature importance vs. removing features"),
    md('''
LightGBM can tell me how much **gain** each column gave its trees. It's tempting to think "high importance =
important feature". But in my experiments I found a column that was used a lot and still didn't help.
'''),
    code(r'''
imp = (lgb_imp / lgb_imp.sum()).sort_values()
fig, ax = plt.subplots(figsize=(9, 7))
top = imp.tail(20)
ax.barh(top.index, top.values, color=[PINK if c.startswith("te_") else PURPLE for c in top.index], height=0.65)
ax.set_title("Top 20 columns by LightGBM gain\n(pink = a target encoding, purple = a raw column)")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
plt.tight_layout(); plt.show()
print(f"share of all gain that comes from target encodings: {imp[[c for c in imp.index if c.startswith('te_')]].sum():.0%}")
'''),
    md('''
## The honest test: take a group away and re-check the score

In an earlier version I added **red-flag features from Joe's fraud-team experience** (number of flags on,
no-authentication × CVV retries, amount ÷ balance, night-time…) and **84 "profile" columns** (each
transaction compared to the average of similar ones). Then I removed one group at a time and re-ran CV
(3 folds, faster settings):

| group removed | change in CV AUC | my decision |
|---|---|---|
| target encodings | **−0.00196** | keep — the most valuable group |
| digit keys | **−0.00148** | keep |
| raw categories | **−0.00117** | keep |
| Joe's red flags | +0.00035 | removing them didn't hurt |
| profiles (84 columns) | +0.00015 | removing them didn't hurt |

The funny part: Joe's `flag_count` was the **4th most-used column** — yet the score did not go down without
it. The trees could already rebuild that information from the six flag columns. And when I re-ran
everything with full settings, the "+0.00035" turned into almost nothing. **Tiny differences like that are
mostly noise** — which brings me to the leaderboard…
'''),

    # ---------------------------------------------------------------- 11
    section(16, "How much should I trust the leaderboard?", "A little statistics about leaderboard noise"),
    md('''
The public leaderboard scores only about **20% of the test set** (~60,000 rows). With fraud this rare, that
is only around a thousand frauds. So I asked: *if my model's true AUC were fixed, how much could the public
score wobble just from which rows happen to be in that 20%?*

I used the **Hanley–McNeil** formula for the standard error of an AUC:

$$SE = \\sqrt{\\frac{A(1-A) + (n_1 - 1)(Q_1 - A^2) + (n_2 - 1)(Q_2 - A^2)}{n_1 n_2}}, \\quad Q_1 = \\frac{A}{2-A},\\; Q_2 = \\frac{2A^2}{1+A}$$

where $A$ is the AUC, $n_1$ the number of frauds and $n_2$ the number of normal transactions.
'''),
    code(r'''
def hanley_mcneil_se(A, n_pos, n_neg):
    q1, q2 = A / (2 - A), 2 * A * A / (1 + A)
    return np.sqrt((A * (1 - A) + (n_pos - 1) * (q1 - A * A) + (n_neg - 1) * (q2 - A * A)) / (n_pos * n_neg))

A = cv["blend"]
n_public = int(round(0.2 * len(test)))
n_pos = int(round(n_public * y.mean()))
se = hanley_mcneil_se(A, n_pos, n_public - n_pos)
print(f"public leaderboard: ~{n_public:,} rows, ~{n_pos:,} frauds")
print(f"standard error of the public AUC: ±{se:.5f}  ->  95% range roughly ±{1.96 * se:.4f}")
print(f"my CV uses all {len(trk):,} rows (~{int(y.sum()):,} frauds): ±{hanley_mcneil_se(A, int(y.sum()), len(trk) - int(y.sum())):.5f}")
'''),
    md('''
That ± looked HUGE to me — bigger than the gap between most teams! But then I realised it answers a
different question: *how far could ONE score land from the "true" score?* When I compare **two of my own
submissions**, they are graded on the **same** rows, and they get mostly the same rows right and wrong. So
the **difference** between them should be much less noisy than each score alone.

I tested this with a **paired bootstrap**: 300 times, I pick a random "fake public leaderboard" of the same
size from my out-of-fold predictions, and score two of my models on it.
'''),
    code(r'''
rng = np.random.default_rng(11)
m1, m2 = "lightgbm", "logistic"
singles, diffs = [], []
for _ in range(300):
    idx = rng.choice(len(y), size=min(n_public, len(y)), replace=True)
    a1, a2 = roc_auc_score(y[idx], oof[m1][idx]), roc_auc_score(y[idx], oof[m2][idx])
    singles.append(a1)
    diffs.append(a1 - a2)
singles, diffs = np.array(singles), np.array(diffs)

fig, axes = plt.subplots(1, 2, figsize=(15, 4))
axes[0].hist(singles, bins=30, color=PURPLE, alpha=0.85)
axes[0].set_title(f"ONE model's score on 300 fake leaderboards\n(spread ±{singles.std():.4f})")
axes[0].set_xlabel("AUC")
axes[1].hist(diffs, bins=30, color=PINK, alpha=0.85)
axes[1].axvline(0, color=INK, lw=1)
axes[1].set_title(f"DIFFERENCE between two models on the same rows\n(spread ±{diffs.std():.4f})")
axes[1].set_xlabel(f"AUC({m1}) − AUC({m2})")
plt.tight_layout(); plt.show()
print(f"one score wobbles by about ±{singles.std():.4f} (the formula said ±{se:.4f})")
print(f"the difference between {m1} and {m2} wobbles by only ±{diffs.std():.4f} "
      f"-> {singles.std() / diffs.std():.1f}x steadier")
print(f"{m1} beat {m2} on {np.mean(diffs > 0):.0%} of the fake leaderboards")
'''),
    note("What this means for my choices 🧠",
         "<b>Comparing my own submissions</b> is much steadier than any single score, but a difference of "
         "0.0001 can still be luck. <b>The whole public board</b> can sit a bit above or below the final private "
         "board, and that shift moves everyone. My CV uses all the training rows, so I choose my final "
         "submissions by <b>CV</b>, and I don't chase tiny leaderboard gains — some top public notebooks are tuned on the public board itself, which may not hold "
         "on the private 80%."),

    # ---------------------------------------------------------------- 12
    section(17, "My experiment log", "Every version, with honest numbers"),
    md('''
I kept every version of my code frozen in its own folder so I could always compare fairly. Here is my
log (CV = my 5-fold cross-validation, LB = public leaderboard):

| version | what changed | CV AUC | public LB |
|---|---|---|---|
| v1 | logistic + LightGBM with 63 leaves, simple frequency features | 0.94595 | 0.94626 |
| v3 | target encoding inside folds, digit keys, **4-leaf** trees with early stopping | 0.95045 | 0.95083 |
| v5 | + joint GLM, "GLM-boost", and **refitting on all rows** for the test predictions | 0.95044 | 0.95146 |
| v6 | fixed a bug in GLM-boost, tuned the GLM's regularization (`C = 0.01` was best) | 0.95049 | — |
| v7 | the public encoding recipe (m-estimate, m = 20), 3-leaf LightGBM averaged over 3 seeds, GLM with splines | 0.95087 | — |
| v8 | + CatBoost and a neural network (trained on our laptop's GPU) | **0.95090** | **0.95161** |
| this notebook | everything above, in one place | *see section 14* | *see the score box above* |

Things I learned from this table:

- 📈 **CV and the leaderboard agree** (within noise), so my CV is trustworthy.
- 🔁 v5 had the **same CV** as v3 but a **better leaderboard** score. The reason: for the test predictions, v5
  trains one last model on **all** the training rows instead of averaging five models that each saw 80%.
  CV can't show that gain, because CV is measured before that last step.
- 🪜 From v1 to v8 my leaderboard score went from 0.94626 to 0.95161 — and every step is in this notebook.
- 🔑 v7 shows how much the *details* of target encoding matter: switching to the m-estimate (m = 20)
  moved my LightGBM from 0.95015 to 0.95062 with the same model.
- 🧪 The GLM regularization sweep was a perfect overfitting curve: too little regularization (`C = 1`) lost
  0.011 AUC because the model memorized noise in tens of thousands of one-hot columns.
'''),
    code(r'''
# (submission.csv was written in section 14)
print("final CV AUC by model:", {n_: round(v, 5) for n_, v in cv.items()})
print(f"whole notebook ran in {(time.time() - T0) / 60:.0f} minutes")
sub.head()
'''),

    # ---------------------------------------------------------------- 13
    section(18, "What I learned & thank-yous", "The short version of this whole notebook"),
    md('''
## 🌷 My thirteen biggest lessons

1. **AUC is about order** — I can compute it by hand by counting pairs.
2. **Accuracy lies** when fraud is rare.
3. **Error bars** keep small groups from fooling me.
4. **Simpson's paradox is real** — always compare inside groups too.
5. **Synthetic data has artifacts** (like the cents!) that wouldn't exist in real life.
6. **PCA doesn't know the target** — variance and fraud signal are different things.
7. **Target encoding is powerful but must be cross-fitted**, or pure noise looks like gold.
8. **Small trees win** when the signal is mostly additive.
9. **"Used a lot" ≠ "helpful"** — test a feature by removing it.
10. **Trust CV over tiny leaderboard differences.**
11. **A straight-line model can be strong** when you give it great inputs (the joint GLM), and trees can
    start from its answer instead of from zero (GLM-boost).
12. **Different beats more of the same** — a blend rewards models that make *different* mistakes.
13. **Train once more on everything** for the final predictions — a gain CV can't even see.

## 🌱 What I want to try next
- a better neural network (embeddings for the categories, several seeds averaged)
- using the original 20,000-row dataset as extra information

## 💌 Thank you
- **alexchilton** — *Credit Card Fraud | Shallow-Tree Stack*: the honest, measured tutorial that taught me the
  most (small trees, digit keys, full refit, leaderboard noise).
- **goodpjw2008** — the digit-key and m-estimate encoding recipe.
- The author of ***Fraud Beneath the Surface*** — the EDA that showed me Simpson's paradox and the
  discovery/confirmation idea.
- **Kaggle** for the Playground Series.
- My teammate and dad **Joe**, who taught me logistic regression, gradient boosting, LightGBM and CatBoost,
  and explained how real fraud teams think.
- I used **Claude** (an AI assistant) as a tutor and to help me write and check my code.

If this notebook helped you or you learned something, an **upvote** would make my day! 💖
And if you spot a mistake, please tell me in the comments — I'm still learning. 🌸
'''),
]


def main():
    nb = {"cells": CELLS,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for c in nb["cells"]:
        c["source"] = c["source"].splitlines(keepends=True)
    out = HERE / "s6e11_showcase.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print("wrote", out, f"({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
