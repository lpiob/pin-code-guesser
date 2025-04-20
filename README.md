Modele algorytmiczne do analizy i predykcji potencjalnych kodów PIN dla zamków
szyfrowych. Działa przy założeniu, że istnieje wiele PINów i że zostały one wygenerowane
przez człowieka, bez wykorzystania generatora liczb losowych.

Modele - gaussowski oraz random forest - sugerują kolejne kody do sprawdzenia. Po ich
dopisaniu do plików `working_codes.txt` i/lub `rejected_codes.txt`, uczą się patternów w
nich zawartych i dokonują kolejnych sugestii.

