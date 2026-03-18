## Movie Success Prediction with Logistic Regression

### Project Overview
This project aims to build a predictive model to determine the success of a movie based on various features such as budget, popularity, runtime, and release date information. We utilize the TMDB 5000 Movie Metadata dataset, which contains information on over 5,000 movies.

### Dataset
The dataset used is the [TMDB 5000 Movie Metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle, consisting of two main files:
- `tmdb_5000_movies.csv`: Contains core movie metadata (budget, genres, homepage, keywords, original language, original title, overview, popularity, production companies, production countries, release date, revenue, runtime, spoken languages, status, tagline, title, vote average, vote count).
- `tmdb_5000_credits.csv`: Contains cast and crew information.

### Methodology
The project follows a standard machine learning pipeline:

1.  **Environment Setup and Data Ingestion**: Installation of necessary libraries (`kaggle`, `opendatasets`, `pandas`) and downloading the dataset.
2.  **Data Exploration**: Initial loading and inspection of the `movies_df` and `credits_df` DataFrames.
3.  **Data Merging and Preprocessing**: 
    -   Merging `movies_df` and `credits_df` on movie ID.
    -   Handling duplicate title columns.
4.  **Feature Engineering**: 
    -   Creation of `is_success` target variable: A movie is considered successful if its revenue is more than 2.5 times its budget.
    -   Extraction of `release_year`, `release_month`, `release_day`, `day_of_week`, and `release_season` from `release_date`.
    -   Calculation of `movie_age` based on the release year.
    -   Dropping irrelevant or noisy columns.
5.  **Handling Missing Values**: 
    -   Treating '0' values in `budget`, `runtime`, `popularity`, and `vote_average` as `NaN`.
    -   Dropping rows with `NaN` values in these critical columns.
    -   Removing redundant columns like `revenue` and `vote_count` after `is_success` is determined.
6.  **Correlation Analysis**: Visualizing the correlation matrix of numerical features to understand relationships and potential multicollinearity.
7.  **Scaling and Standardization**: Applying `StandardScaler` to features (`X`) to normalize their range, which is crucial for Logistic Regression.
8.  **Model Training and Evaluation**: 
    -   Splitting the data into training and testing sets (80/20 split).
    -   Training a `LogisticRegression` model with `class_weight='balanced'` to handle potential class imbalance.
    -   Evaluating the model using `accuracy_score`, `classification_report`, `confusion_matrix`, and `ROC curve`.
    -   Analyzing feature importance using the absolute coefficients of the logistic regression model.

### Results
The Logistic Regression model achieved an accuracy of approximately **74%**. Key evaluation metrics from the classification report, confusion matrix, and ROC curve provide insight into the model's performance in predicting movie success (precision, recall, f1-score, AUC).

### Inference Function
The `predict_movie_success` and `test_new_movie` functions allow for interactive prediction of a new movie's success based on user-provided parameters like budget, popularity, runtime, vote average, release month, and movie age.
