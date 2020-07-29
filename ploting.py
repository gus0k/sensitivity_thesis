import dill
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt as plt

from pathlib import Path

with open('stats_sensitivity_rolling', 'rb') as fh:
    rolling = dill.load(fh)

receding_selling = []
receding_keeping = []
for fl in Path('results_receding/').glob('*'):
    with open(fl, 'rb') as fh:
        d = dill.load(fh)
        receding_selling.append(d[0][0])
        receding_keeping.append(d[1][0])






