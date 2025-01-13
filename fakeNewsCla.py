# Loading essential libraries
import numpy as np
import pandas as pd
# Importing essential libraries for performing Natural Language Processing on 'kaggle_fake_train' dataset
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Hyperparameter tuning
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import pickle

## import data
df = pd.read_csv('kaggle_fake_train.csv')

# Dropping the 'id' column
df.drop('id', axis=1, inplace=True)

# Dropping NaN values as the NaN values are less compared to records
df.dropna(inplace=True)

news = df.copy()

news.reset_index(inplace=True)

# Cleaning the news
corpus = []
ps = PorterStemmer()

for i in range(0,news.shape[0]):

  # Cleaning special character from the news-title
  title = re.sub(pattern='[^a-zA-Z]', repl=' ', string=news.title[i])

  # Converting the entire news-title to lower case
  title = title.lower()

  # Tokenizing the news-title by words
  words = title.split()

  # Removing the stopwords
  words = [word for word in words if word not in set(stopwords.words('english'))]

  # Stemming the words
  words = [ps.stem(word) for word in words]

  # Joining the stemmed words
  title = ' '.join(words)

  # Building a corpus of news-title
  corpus.append(title)


# Creating the Bag of Words model
CV = CountVectorizer(max_features=5000, ngram_range=(1,3))
X = CV.fit_transform(corpus).toarray()

# Extracting dependent variable from the dataset
y = news['label']


## Model Building
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

## Logistic Regression Model
lr_classifier = LogisticRegression(random_state=0)


# Hyperparameter Grid
lr_params = {
    'C': [0.1, 0.3, 0.5, 0.7, 0.8, 1.0, 5, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# K-Fold Cross Validation Setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV Configuration
lr_grid_search = GridSearchCV(
    lr_classifier,  
    lr_params, 
    cv=cv, 
    scoring='accuracy', 
    n_jobs=-1,  
    verbose=1
)

# Fit GridSearchCV
lr_grid_search.fit(X_train, y_train)

# Print best parameters
print("Best Parameters:", lr_grid_search.best_params_)
print("Best Score:", lr_grid_search.best_score_)


# Retraining model with best params
best_lr_classifier = LogisticRegression(
    C=lr_grid_search.best_params_['C'],
    penalty=lr_grid_search.best_params_['penalty'],
    solver=lr_grid_search.best_params_['solver'],
    random_state=0
)

#Fit the model
best_lr_classifier.fit(X_train,y_train)

## Saving model to disk
pickle.dump(best_lr_classifier,open('model.pkl','wb'))
pickle.dump(CV, open('count_vectorizer.pkl', 'wb'))