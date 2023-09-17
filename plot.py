import matplotlib.pyplot as plt
from data import get_data

field = 'SalePriceLog'
df = get_data()
df.hist(bins=50, figsize=(12, 8))
plt.show()
