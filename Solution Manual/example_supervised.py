import os

import matplotlib.pyplot as plt
import pandas as pd
from feature_ranking import feature_selection_supervised
from feature_ranking_pdf import create_report

# ====================================================================================
# dir path
file_path = os.getcwd() + "/Feature Ranking/Auxiliary Files/"
result_path = os.getcwd() + "/Feature Ranking/Result Files/"

# read the data
df = pd.read_csv(os.path.join(file_path, "seeds_dataset.csv"))

# separate input and output file
df_input = df.iloc[:, 0:-1]
df_output = df.iloc[:, -1]

# call ranking feature function
df_result = feature_selection_supervised(df_input, df_output, "nominal", True)

# plot/save the result as a graph
df_result.T.plot.bar(colormap="Set2", rot=45)
plt.ylabel("Score")
plt.tight_layout()
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.savefig(result_path + "plot.png", bbox_inches="tight", transparent=True)
# create a report
create_report(df_result, result_path)
# print the result dataframe
print(df_result)
