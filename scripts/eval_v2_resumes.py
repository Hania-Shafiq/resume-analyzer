import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
from app.parser import extract_text
from app.preprocess import light_clean

def main():
    bundle = joblib.load(PROJECT_ROOT / 'app/models/job_classifier_v2.joblib')
    estimator = bundle['estimator']
    classes = estimator.classes_

    resume_dir = PROJECT_ROOT / 'tests/eval/resumes'
    resume_files = sorted(resume_dir.glob('*.*'))
    print(f'Total resume files in {resume_dir}: {len(resume_files)}')

    results_15 = []
    results_all = []

    for idx, p in enumerate(resume_files, 1):
        raw_text = extract_text(p)
        cleaned = light_clean(raw_text)
        probs = estimator.predict_proba([cleaned])[0]
        top_i = int(np.argmax(probs))
        top_prob = float(probs[top_i])
        top_role = str(classes[top_i])
        status = 'uncertain' if top_prob < 0.40 else 'confident'
        entry = {
            'index': idx,
            'filename': p.name,
            'top_role': top_role,
            'top_prob': top_prob,
            'top_pct': round(top_prob * 100, 2),
            'status': status
        }
        if idx <= 15:
            results_15.append(entry)
        results_all.append(entry)

    print('\nEvaluation on first 15 test resumes (threshold < 0.40):')
    print('-' * 85)
    print(f'{"#":<3} {"Filename":<32} {"Top Role":<25} {"Prob %":<10} {"Status"}')
    print('-' * 85)
    for r in results_15:
        print(f"{r['index']:<3} {r['filename']:<32} {r['top_role']:<25} {r['top_pct']:<10.2f} {r['status']}")

    uncertain_15 = [r for r in results_15 if r['status'] == 'uncertain']
    print('-' * 85)
    print(f'Total falling under threshold (< 0.40) among first 15: {len(uncertain_15)} / 15')
    if uncertain_15:
        print('Uncertain resumes:')
        for u in uncertain_15:
            print(f"  - {u['filename']}: {u['top_role']} ({u['top_pct']}%)")
    else:
        print('None of the first 15 resumes fell under 0.40.')

    print('\n' + '=' * 85)
    print('Evaluation across all 20 test resumes:')
    print('-' * 85)
    for r in results_all:
        print(f"{r['index']:<3} {r['filename']:<32} {r['top_role']:<25} {r['top_pct']:<10.2f} {r['status']}")

    uncertain_all = [r for r in results_all if r['status'] == 'uncertain']
    print('-' * 85)
    print(f'Total falling under threshold (< 0.40) among all 20: {len(uncertain_all)} / {len(results_all)}')

if __name__ == '__main__':
    main()
