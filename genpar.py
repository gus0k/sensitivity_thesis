IDS = range(126)
STARTS = [10, 20, 30, 40, 50]

for id_ in IDS:
    for st in STARTS:
        string = ' '.join(map(str, [id_, st]))
        print(string)
