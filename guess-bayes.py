import os
import numpy as np
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

WORKING_FILE = "working_codes.txt"
REJECTED_FILE = "rejected_codes.txt"
SUGGEST_COUNT = 4

def read_codes(filename):
    if not os.path.exists(filename):
        return []
    with open(filename, "r") as f:
        return [code.strip() for code in f if code.strip().isdigit() and len(code.strip()) == 4]

def encode_code(code):
    return [int(d) for d in code]

def main():
    working = read_codes(WORKING_FILE)
    rejected = read_codes(REJECTED_FILE)

    if len(working) == 0 or len(rejected) == 0:
        print("Both code files required")
        return

    X = np.array([encode_code(c) for c in working + rejected])
    y = np.array([0.0 if c in rejected else 1.0 for c in working + rejected])

    # Przestrzeń kodów: 4 cyfry od 0 do 9
    space = [Integer(0, 9, name=f"d{i}") for i in range(4)]

    @use_named_args(space)
    def objective(**kwargs):
        code = [kwargs[f"d{i}"] for i in range(4)]
        for known_code, label in zip(X, y):
            if list(code) == list(known_code):
                return 1 - label  # 0 dla działającego, 1 dla niedziałającego
        return 0.5  # nieznany kod – model powinien się tego nauczyć

    res = gp_minimize(
        func=objective,
        dimensions=space,
        acq_func="EI",
        n_calls=100,
        n_initial_points=10,
        random_state=42,
        verbose=False,
    )

    # Propozycje nowych kodów, które są nieznane
    tried = set("".join(map(str, x)) for x in X)
    suggestions = []
    for x in res.x_iters[::-1]:
        code = "".join(str(d) for d in x)
        if code not in tried and code not in suggestions:
            suggestions.append(code)
        if len(suggestions) >= SUGGEST_COUNT:
            break

    for code in suggestions:
        print(code)

if __name__ == "__main__":
    main()

