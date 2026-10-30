# Amazon Review Sentiment: Traditional ML and BERT

A binary sentiment-classification project comparing TF-IDF baselines with optional BERT fine-tuning. Ratings 1–2 are mapped to negative, ratings 4–5 to positive, and neutral rating-3 reviews are excluded. This is a rating-derived label rule, not an independent human sentiment annotation.

## Project files

- [`notebooks/amazon_review_sentiment.ipynb`](notebooks/amazon_review_sentiment.ipynb): data checks, exploratory analysis, shared split, traditional baselines, optional BERT training, and final evaluation.
- `requirements.txt`: Python dependencies.

## Dataset

Download `amazon_reviews.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/henryshan/amazon-review-sentiment-analysis) and place it in `data/` or `/content/` for Google Colab. The CSV is not included in this repository; review the dataset's license and terms.

The supplied CSV contains 4,915 rows and 12 columns. After removing the one row with missing review text and omitting neutral three-star reviews, 4,772 reviews remain: 4,448 positive (93.2%) and 324 negative (6.8%). The resulting class imbalance makes accuracy alone a weak evaluation measure.

## Run

1. Install dependencies with `pip install -r requirements.txt`.
2. Download the CSV from Kaggle and place it in the expected path.
3. Run the notebook from top to bottom in Jupyter or Colab.
4. TF-IDF baselines are the lightweight path. BERT fine-tuning is optional and is best run with a GPU.

## Method

All models use a shared stratified train/validation/test split. TF-IDF is fitted on training text only, and the held-out test split is not used for model selection or early stopping. The notebook uses class-weighted linear baselines rather than applying SMOTE to sparse text vectors. Compare macro F1 and negative-class recall alongside accuracy.

## Results status

The data profile above was checked against the supplied CSV. Model metrics for the corrected shared-split workflow have not yet been run; the prior HTML report's scores are not carried forward because its BERT training used the test split as validation and the saved run was incomplete. Run the notebook before publishing model scores.

## Limitations

Star ratings are an imperfect proxy for the sentiment expressed in review text, and excluding neutral reviews changes the task. Results may not generalize to other product categories, platforms, or time periods.

## LinkedIn project description

**Amazon Review Sentiment Classification** — Prepared a rating-derived binary sentiment dataset and designed a comparison of TF-IDF Logistic Regression, Linear SVM, Naive Bayes, and BERT. Used stratified train/validation/test splits and planned evaluation with macro F1 and minority-class recall to account for strong label imbalance.
