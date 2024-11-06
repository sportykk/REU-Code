# This is a temp file for fixing the dataset 

# This is code is sitting aside for a moment
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppress the oneDNN custom operations message


import pandas as pd

# original_df = pd.read_csv('eclipse_platform_updated.csv')
# smaller_df = pd.read_csv('test_documents_Platform.csv')

# columns_to_add = ['Created_time', 'Resolved_time'] 

# common_columns = list(set(smaller_df.columns) & set(original_df.columns))

# merged_df = pd.merge(smaller_df, original_df[common_columns + columns_to_add], on=common_columns, how='left')

# merged_df.to_csv('merged_platform.csv', index=False)
# print("Merge completed. Result saved to 'merged_platform.csv'")

# Load the dataset
data = pd.read_csv('merged_platform.csv')
# fixed_bugs = data[data['Resolution'] == 'FIXED']
# fixed_bugs = fixed_bugs[fixed_bugs['Assignee Real Name'].notna()]
# devNum = fixed_bugs['Assignee Real Name'].value_counts()

#devNum = devNum[(devNum > 220) & (~devNum.index.str.startswith('Platform'))]

# devNum = devNum[(devNum > 600) & (~devNum.index.str.startswith('Platform')) & (~devNum.index.str.startswith('platform'))]
# fixed_bugs = fixed_bugs[fixed_bugs['Assignee Real Name'].isin(devNum.index)]

# print(devNum)
# print(len(devNum))
# print(len(fixed_bugs))

print(len(data))

# import torch
# print("CUDA Available:", torch.cuda.is_available())
# print("Number of GPUs:", torch.cuda.device_count())
# print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU Found")

