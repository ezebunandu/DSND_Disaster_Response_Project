import pickle
import sys

import nltk
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

nltk.download(["punkt", "wordnet"])


def load_data(database_filepath):
    """Connects to the database at database_filepath and
    reads in the data from the messages table
    returns the message column as X, the labels as Y,
    and a list of the various categories of the labels
    """
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql("SELECT * FROM messages", engine)
    X = df["message"]
    Y = df.drop(["id", "message", "original", "genre"], axis=1)
    category_names = Y.columns.to_list()
    return X, Y, category_names


def tokenize(text):
    """Tokenizes the text by first using nltk word_tokenize and
    then lemmatizing the words in the text"""
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(token).strip() for token in tokens]
    return clean_tokens


def build_model():
    model = Pipeline(
        [
            (
                "text_pipeline",
                Pipeline(
                    [
                        ("vect", CountVectorizer(tokenizer=tokenize)),
                        ("tfidf", TfidfTransformer()),
                    ]
                ),
            ),
            ("clf", MultiOutputClassifier(RandomForestClassifier(random_state=42))),
        ]
    )
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    Y_test_pred = model.predict(X_test)
    Y_test_pred_df = pd.DataFrame(Y_test_pred)
    for i in range(Y_test_pred_df.shape[1]):
        print(f"\033[1m{category_names[i]}\033[0m")
        print(
            classification_report(
                Y_test.iloc[:, i], Y_test_pred_df.iloc[:, i], zero_division=0
            ),
            end="",
        )
        print("--------------------------------------------------------")
        print()


def save_model(model, model_filepath):
    with open(model_filepath, "wb") as fin:
        pickle.dump(model, fin)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        parameters = {'text_pipeline__vect__ngram_range': ((1, 2), ),
                      'text_pipeline__vect__max_df': (0.75, ),
                      'text_pipeline__vect__max_features': (5000,),
                      'clf__estimator__n_estimators': [200, 400]
        }

        model = GridSearchCV(model, param_grid=parameters, n_jobs=-1)

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
