SEEDS = [11, 22, 33, 44]
IDS = range(126)
STARTS = [10, 20, 30, 40]

for se in SEEDS:
    for id_ in IDS:
        for st in STARTS:
            string = ' '.join(map(str, [se, id_, st]))
            print(string)
