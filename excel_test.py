import numpy as np
import pandas as pd
import os

os.mkdir("test")

test_data = np.array([[1], [2], [3], [4], [5]])
test_df = pd.DataFrame(data=[[x, x, x] for x in range(3)], columns=["a", "b", "c"])
test_df.to_excel("test/test_excel.xlsx", index=False)
print(test_df)
