import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# key position map
keypad_pos = {
    '1': (0, 0), '2': (0, 1), '3': (0, 2),
    '4': (1, 0), '5': (1, 1), '6': (1, 2),
    '7': (2, 0), '8': (2, 1), '9': (2, 2),
    '0': (3, 1)
}

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

sequential_codes = set()
for start in range(10):
    seq = ''.join(str((start + i) % 10) for i in range(4))
    sequential_codes.add(seq)
    sequential_codes.add(seq[::-1])

all_codes = [f"{i:04d}" for i in range(10000)]

def load_known_codes():
    with open("working_codes.txt") as f:
        good = [line.strip() for line in f if line.strip()]
    with open("rejected_codes.txt") as f:
        bad = [line.strip() for line in f if line.strip()]
    return good, bad

def build_dataset(good, bad):
    data = []
    for code in good:
        row = extract_features(code)
        row['label'] = 1
        row['code'] = code
        data.append(row)
    for code in bad:
        row = extract_features(code)
        row['label'] = 0
        row['code'] = code
        data.append(row)
    return pd.DataFrame(data)

def suggest_next(df, tried_codes):
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    X = df.drop(columns=["label", "code"])
    y = df["label"]
    clf.fit(X, y)
    show_feature_importance(clf, X.columns)

    candidates = [code for code in all_codes if code not in tried_codes]
    features = pd.DataFrame([extract_features(code) for code in candidates])
    features["code"] = candidates
    features["prob"] = clf.predict_proba(features.drop(columns=["code"]))[:, 1]
    top = features.sort_values("prob", ascending=False).head(4)
    return top[["code", "prob"]]

def show_feature_importance(model, feature_names, top_n=50):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    print("\nTop feature importances:")
    for i in sorted_idx:
        print(f"{feature_names[i]:<20} {importances[i]:.4f}")
    print("\n")

def main():
    good, bad = load_known_codes()
    tried = set(good + bad)
    df = build_dataset(good, bad)
    suggestions = suggest_next(df, tried)
    print("Top 4 suggested codes:")
    print(suggestions.to_string(index=False))

if __name__ == "__main__":
    main()

