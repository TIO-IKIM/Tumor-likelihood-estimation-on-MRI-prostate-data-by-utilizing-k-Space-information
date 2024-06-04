import pandas as pd
from pathlib import Path

data_csv = pd.read_csv("dwi_slice_level_labels.csv")

data_csv.drop(columns=['Unnamed: 0'], inplace=True)
data_csv.drop(columns=['folder'], inplace=True)
data_csv['fastmri_rawfile'] = data_csv['fastmri_rawfile'].str.replace('file_prostate_AXDIFF_', 'file_prostate_AXDIFF_0')

# Split the data_csv into training, validation and test, according to the "data_split" column

train_data = data_csv[data_csv['data_split'] == 'training']
val_data = data_csv[data_csv['data_split'] == 'validation']
test_data = data_csv[data_csv['data_split'] == 'test']

# For each slice in train_data in the column "slice" change the name of the file in "fastmri_rawfile" to [file]_[slice]

for index, row in train_data.iterrows():
    train_data.at[index, 'fastmri_rawfile'] = Path(row['fastmri_rawfile']).stem + '_' + str(f"slice{row['slice']}") + '.npy'

for index, row in val_data.iterrows():
    val_data.at[index, 'fastmri_rawfile'] = Path(row['fastmri_rawfile']).stem + '_' + str(f"slice{row['slice']}") + '.npy'

for index, row in test_data.iterrows():
    test_data.at[index, 'fastmri_rawfile'] = Path(row['fastmri_rawfile']).stem + '_' + str(f"slice{row['slice']}") + '.npy'

train_data.to_csv("dwi_2D_train.csv", index=False)
val_data.to_csv("dwi_2D_val.csv", index=False)
test_data.to_csv("dwi_2D_test.csv", index=False)