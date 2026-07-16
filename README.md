# Apple Products Pricing Analysis (2020–2026)

## TL;DR

This project analyzes 80,000 price records of Apple products (iPhone, iPad, Mac, Watch — 31 models) on **Amazon and Flipkart India** from Sep 2020 to Jul 2026, answering when to buy, where to buy, and what a "fair price" looks like. A Random Forest model predicts market price with **MAE ≈ $23 (R² 0.99)**, enabling automatic detection of unusually good deals.

📓 Full analysis: [`apple_pricing_analysis.ipynb`](apple_pricing_analysis.ipynb)

## Business Problem

Apple products depreciate on a predictable yearly release cycle, and prices swing heavily around sale events. Different stakeholders need different answers from the same data:

| Stakeholder | Question |
|---|---|
| Consumers | When and where is the best time/platform to buy? |
| Retailers | How deep must event discounts be? Does discounting drain stock? |
| Resellers / refurbishers | How fast do products depreciate? What margin does refurbished offer? |
| Brand analysts | Does heavy discounting hurt ratings / brand perception? |

## Hypotheses Tested (α = 0.05)

| # | Hypothesis | Test |
|---|---|---|
| H1 | Sale events carry deeper discounts than regular days | Welch's t-test |
| H2 | Discount depth differs across sale event types (Black Friday, Prime Day, Big Billion Days, Great Indian Festival) | One-way ANOVA |
| H3 | Amazon and Flipkart discount differently | Welch's t-test |
| H4 | Refurbished units are discounted more than new | Welch's t-test |
| H5 | Deep discounts are associated with different ratings | Welch's t-test |

Full test statistics, p-values, and decisions are in Section 13 of the notebook. Note: with n = 80,000, statistical significance is easy to reach — results are interpreted against group means for **practical** significance.

## What Can Be Predicted

A **"fair price" engine**: given launch price, days since launch, platform, category, condition, stock status, and whether a sale is running, the model predicts the expected current market price.

| Model | MAE (USD) | R² |
|---|---|---|
| Linear Regression (baseline) | 57.01 | 0.969 |
| **Random Forest** | **23.12** | **0.993** |

Validation uses a **time-based split** (train < Jul 2025, test = last 12 months) to avoid leakage. Launch price and days-since-launch dominate feature importance — depreciation is strongly non-linear, which is why the Random Forest beats the linear baseline.

**Business application:** the residual (predicted fair price − actual price) works as a *deal score*. Listings priced far below the model's estimate are flagged as the best buying opportunities — the notebook ranks the top 10 deals in the test period.

## Data

`apple_products_pricing_2020_2026.csv` (not tracked in this repo — see `.gitignore`): 80,000 rows × 14 columns covering date, platform, model, condition, launch/current prices (USD & INR), discount %, sale event, stock status, rating, and review count.

## Reproducing

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
jupyter notebook apple_pricing_analysis.ipynb
```

Place the CSV in the repo root before running.
