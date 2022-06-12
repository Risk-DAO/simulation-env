# Stability Report

### Setup

```bash
# Minimum requirement 
Python3.7
sudo apt install python3.7 python3.7-venv
 
# Install dependencies:
datetime
glob
math
sys
matplotlib.pyplot as plt
pandas as pd
numpy as np
copy
seaborn as sns
stability_pool_simple
joblib import Parallel, delayed
multiprocessing
uuid
shutil
os
```

### Config

```bash
    "tri": {
        'series_std_ratio': 0.66,
        'volume_for_slippage_10_percentss': [750, 1500],
        'trade_every': 1800,
        'collaterals': [5_000_000 / ETH_PRICE, 10_000_000 / ETH_PRICE, 15_000_000 / ETH_PRICE, 30_000_000 / ETH_PRICE],
        'liquidation_incentives': [0.1],
        'stability_pool_initial_balances': [0, 0.1, 0.25, 0.5],
        'share_institutionals': [0, 0.25, 0.5],
        'recovery_halflife_retails': [1, 5, 10],
        'price_recovery_times': [0.000001],
        'l_factors': [0.5, 1, 2, 4, 6]
    }
```

### Run
```bash
python stability_report.py
``` 
