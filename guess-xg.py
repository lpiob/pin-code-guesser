import xgboost as xgb
import pandas as pd
import numpy as np

keypad_pos = {
    '1': (0, 0), '2': (0, 1), '3': (0, 2),
    '4': (1, 0), '5': (1, 1), '6': (1, 2),
    '7': (2, 0), '8': (2, 1), '9': (2, 2),
    '0': (3, 1)
}

sequential_codes = set()
for start in range(10):
    seq = ''.join(str((start + i) % 10) for i in range(4))
    sequential_codes.add(seq)
    sequential_codes.add(seq[::-1])

def extract_features(code):
    digits = [int(d) for d in code]
    coords = [keypad_pos[d] for d in code]
    keypad_dist = sum(
        ((coords[i][0] - coords[i-1][0])**2 + (coords[i][1] - coords[i-1][1])**2)**0.5
        for i in range(1, 4)
    )

    col_counts = [0, 0, 0]  # columns 0, 1, 2
    for d in code:
        col = keypad_pos[d][1]
        col_counts[col] += 1

    row_counts = [0, 0, 0, 0]
    for d in code:
        row = keypad_pos[d][0]
        row_counts[row] += 1

    return {
        'd0': digits[0],
        'd1': digits[1],
        'd2': digits[2],
        'd3': digits[3],
        'col_0_count': col_counts[0],
        'col_1_count': col_counts[1],
        'col_2_count': col_counts[2],
        'row_0_count': row_counts[0],
        'row_1_count': row_counts[1],
        'row_2_count': row_counts[2],
        'row_3_count': row_counts[3],
        'has_0_within': int('0' in code), 
        'has_1_within': int('1' in code),
        'has_2_within': int('2' in code), 
        'has_3_within': int('3' in code),
        'has_4_within': int('4' in code), 
        'has_5_within': int('5' in code),
        'has_6_within': int('6' in code), 
        'has_7_within': int('7' in code),
        'has_8_within': int('8' in code), 
        'has_9_within': int('9' in code),
        'sum': sum(digits),
        'unique_digits': len(set(digits)),
        'has_repeats': int(len(set(digits)) < 4),
        'is_sequential': int(code in sequential_codes),
        'looks_like_year': int(1900 <= int(code) <= 2099),
        'is_palindrome': int(code == code[::-1]),
        'is_xyxy': int(digits[0] == digits[2] and digits[1] == digits[3] and digits[0] != digits[1]),
        'is_xxyy': int(digits[0] == digits[1] and digits[2] == digits[3] and digits[0] != digits[2]),
        'is_xyyx': int(digits[0] == digits[3] and digits[1] == digits[2] and digits[0] != digits[1]),
        'all_even': int(all(d % 2 == 0 for d in digits)),
        'all_odd': int(all(d % 2 == 1 for d in digits)),
        'starts_with_zero': int(code.startswith('0')),
        'ends_with_zero': int(code.endswith('0')),
        'has_triplet': int(any(digits.count(d) == 3 for d in set(digits))),
        'is_ascending': int(digits == sorted(digits)),
        'is_descending': int(digits == sorted(digits, reverse=True)),
        'keypad_path_length': keypad_dist
    }

def load_data():
    with open('working_codes.txt') as f:
        pos = [line.strip() for line in f if line.strip()]
    with open('rejected_codes.txt') as f:
        neg = [line.strip() for line in f if line.strip()]

    all_samples = [(p, 1) for p in pos] + [(n, 0) for n in neg]
    X = pd.DataFrame([extract_features(p) for p, _ in all_samples])
    y = np.array([label for _, label in all_samples])
    return X, y, set(pos + neg)

def train_model(X, y):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    return model

def generate_all_pins():
    return [str(i).zfill(4) for i in range(10000)]

def guess_next(model, tried_set, X_train):
    candidates = [p for p in generate_all_pins() if p not in tried_set]
    X_candidates = pd.DataFrame([extract_features(p) for p in candidates])
    probs = model.predict_proba(X_candidates)[:, 1]
    ranked = sorted(zip(candidates, probs), key=lambda x: x[1], reverse=True)

    importances = model.feature_importances_
    feature_names = X_train.columns
    sorted_idx = np.argsort(importances)[::-1][:10]
    print("\nTop feature importances:")
    for i in sorted_idx:
        print(f"{feature_names[i]:<25} {importances[i]:.4f}")
    print()

    return [code for code, _ in ranked[:4]]

if __name__ == '__main__':
    X, y, tried = load_data()
    model = train_model(X, y)
    next_guesses = guess_next(model, tried, X)
    print("Next codes to try:")
    for code in next_guesses:
        print(code)

