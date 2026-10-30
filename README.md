# Amazon Review Sentiment: Traditional ML and BERT

A binary sentiment-classification comparison using TF-IDF baselines and BERT. Ratings 1–2 are mapped to negative, 4–5 to positive, and neutral rating-3 reviews are excluded.

## Project files

- [`notebooks/amazon_review_sentiment.ipynb`](notebooks/amazon_review_sentiment.ipynb): cleaned analysis, shared data split, traditional baselines, optional BERT fine-tuning, and final evaluation.
- `requirements.txt`: Python dependencies.

## Dataset

Download `amazon_reviews.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/henryshan/amazon-review-sentiment-analysis) and place it in `data/` or `/content/` for Colab. The dataset is not stored in this repository. Review the dataset's license and terms.

The notebook requires the columns `reviewText` and `overall`. It uses only ratings 1, 2, 4, and 5; rating 3 is omitted.

## Run

1. Install dependencies with `pip install -r requirements.txt`.
2. Download the CSV from Kaggle and place it in the expected path.
3. Run the notebook top to bottom.
4. The classical baselines run first. BERT fine-tuning is optional and is best run with a GPU.

## Method notes

All models share a stratified train, validation, and test split. TF-IDF is fitted only on training data. Validation is used for model selection; the test split is held back for final comparison. Class-weighted linear models are used instead of SMOTE on sparse text vectors. Compare macro F1 and negative-class recall as well as accuracy because the original data is strongly imbalanced.

## Results

The previous HTML report recorded model scores, but its BERT training used the test set as validation and the supplied notebook contains an incomplete BERT run. Those historic scores are not reported as results for this corrected workflow. Populate this section after a complete run of the cleaned notebook.

## Limitations

Star ratings are an imperfect proxy for text sentiment, and excluding neutral ratings changes the task. Results may not generalize to other products, platforms, or time periods.

## LinkedIn project description

**Amazon Review Sentiment Classification** — Compared TF-IDF-based Logistic Regression, Linear SVM, and Naive Bayes baselines with BERT on Amazon review text. Used stratified train/validation/test splits, addressed class imbalance, and evaluated models with macro F1 and minority-class recall. Add verified metrics after rerunning the cleaned notebook.
