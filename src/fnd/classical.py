"""TF-IDF models, each tuned on the validation split (F1 on the Fake class)."""
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def make_vectorizer():
    return TfidfVectorizer(stop_words='english', max_df=0.8, min_df=5, ngram_range=(1, 2),
                           max_features=60000, sublinear_tf=True)


# name -> (factory(seed, **params), grid)
MODELS = {
    'Naive Bayes': (lambda seed, **p: MultinomialNB(**p),
                    {'alpha': [0.01, 0.03, 0.1, 0.3, 1.0]}),
    'Logistic Regression': (lambda seed, **p: LogisticRegression(max_iter=2000, class_weight='balanced',
                                                                 random_state=seed, **p),
                            {'C': [0.3, 1, 3, 10, 30]}),
    'Linear SVM': (lambda seed, **p: LinearSVC(class_weight='balanced', random_state=seed, **p),
                   {'C': [0.03, 0.1, 0.3, 1, 3]}),
    'Decision Tree': (lambda seed, **p: DecisionTreeClassifier(random_state=seed, **p),
                      {'max_depth': [30, 60], 'min_samples_leaf': [1, 5]}),
    'Random Forest': (lambda seed, **p: RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                                               random_state=seed, **p),
                      {'max_features': ['sqrt', 0.02], 'min_samples_leaf': [1, 2]}),
    # n_estimators is chosen by early stopping on the validation split
    'XGBoost': (lambda seed, **p: XGBClassifier(n_estimators=1000, learning_rate=0.1, subsample=0.9,
                                                colsample_bytree=0.5, tree_method='hist',
                                                eval_metric='logloss', early_stopping_rounds=30,
                                                n_jobs=-1, random_state=seed, **p),
                {'max_depth': [4, 8]}),
}


def scores(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _fit(name, model, X_tr, y_tr, X_val, y_val):
    if name == 'XGBoost':
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_tr, y_tr)
    return model


def fit_tuned(name, X_tr, y_tr, X_val, y_val, seed, params=None):
    """Fit `name`; grid-search on the validation split unless `params` is given.

    Returns (model, chosen params, val F1, train seconds of the final fit).
    """
    factory, grid = MODELS[name]
    candidates = [params] if params is not None else list(ParameterGrid(grid))
    best = None
    for p in candidates:
        start = time.perf_counter()
        model = _fit(name, factory(seed, **p), X_tr, y_tr, X_val, y_val)
        seconds = time.perf_counter() - start
        val_f1 = f1_score(y_val, model.predict(X_val))
        if best is None or val_f1 > best[2]:
            best = (model, p, val_f1, seconds)
    model, p, val_f1, seconds = best
    if name == 'XGBoost':
        p = {**p, 'best_iteration': int(model.best_iteration)}
    return model, p, val_f1, seconds


def strip_fitted_only(params):
    """Drop fitted-only entries (e.g. XGBoost best_iteration) before reusing params."""
    return {k: v for k, v in params.items() if k != 'best_iteration'}
