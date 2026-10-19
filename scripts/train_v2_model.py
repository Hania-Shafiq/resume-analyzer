import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from app.preprocess import light_clean

RANDOM_STATE = 42

def main():
    # 1. Load original dataset
    df_orig = pd.read_csv('app/data/resume_dataset.csv')
    print(f'Original dataset rows: {len(df_orig)}')

    # 2. Load supplement
    supp_dir = Path('data/devops_supplement')
    supp_rows = []
    for p in sorted(supp_dir.glob('*.txt')):
        text = p.read_text(encoding='utf-8').strip()
        supp_rows.append({'Category': 'DevOps / Cloud Engineer', 'Resume_str': text})
    df_supp = pd.DataFrame(supp_rows)
    print(f'Supplement rows: {len(df_supp)}')

    df_all = pd.concat([df_orig, df_supp], ignore_index=True)
    num_classes = df_all['Category'].nunique()
    print(f'Total rows: {len(df_all)}, Classes: {num_classes}')

    # 3. Clean text
    df_all['clean_text'] = df_all['Resume_str'].apply(light_clean)

    # 4. Stratified split
    X = df_all['clean_text']
    y = df_all['Category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f'Train set: {len(X_train)} | Test set: {len(X_test)}')
    devops_train = (y_train == 'DevOps / Cloud Engineer').sum()
    devops_test = (y_test == 'DevOps / Cloud Engineer').sum()
    print(f'DevOps in train: {devops_train} | in test: {devops_test}')

    # 5. Pipeline & Grid Search
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tfidf_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            token_pattern=r'(?u)\b\w[\w+#]*',
        )),
        ('clf', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])

    SCORING = {'f1_macro': 'f1_macro', 'neg_log_loss': 'neg_log_loss'}
    grid = GridSearchCV(
        tfidf_pipe,
        param_grid={'clf__C': [1, 10, 100]},
        cv=cv,
        scoring=SCORING,
        refit='neg_log_loss',
        n_jobs=1,
    )
    grid.fit(X_train, y_train)

    # Evaluate on test set
    best_C = grid.best_params_['clf__C']
    pred_test = grid.predict(X_test)
    proba_test = grid.predict_proba(X_test)

    test_acc = accuracy_score(y_test, pred_test)
    test_f1 = f1_score(y_test, pred_test, average='macro')
    cv_f1 = grid.cv_results_['mean_test_f1_macro'][grid.best_index_]

    print(f'Best C: {best_C}')
    print(f'CV Macro-F1: {cv_f1:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Test Macro-F1: {test_f1:.4f}')
    print('\nClassification Report (Test Set):')
    print(classification_report(y_test, pred_test, digits=4))

    # Final refit on ALL data
    final_estimator = clone(grid.best_estimator_).fit(X, y)

    # Save bundle
    save_path = Path('app/models/job_classifier_v2.joblib')
    save_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        'kind': 'tfidf',
        'model_name': 'TF-IDF (1-2 ngrams) + Logistic Regression v2',
        'estimator': final_estimator,
        'classes': [str(c) for c in final_estimator.classes_],
        'metrics': {
            'cv_macro_f1': round(float(cv_f1), 4),
            'test_accuracy': round(float(test_acc), 4),
            'test_macro_f1': round(float(test_f1), 4),
        },
        'trained_on': int(len(df_all)),
        'sklearn_version': sklearn.__version__,
        'embedding_model': None,
    }
    joblib.dump(bundle, save_path, compress=3)
    print(f'Saved -> {save_path} ({save_path.stat().st_size / 1024:.1f} KB)')
    print('Classes:', bundle['classes'])

if __name__ == '__main__':
    main()
