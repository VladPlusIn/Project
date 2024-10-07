import pandas as pd
from sklearn.utils import shuffle

# Define paths
output_folder = '/home/vladplyusnin/tftest/Deep-Learning-COPSCI764/Project/data/ipinyou/total/'

# Load the merged train and test datasets, set low_memory=False to avoid DtypeWarning
train_file = output_folder + 'train.log.txt'
test_file = output_folder + 'test.log.txt'

train_data = pd.read_csv(train_file, sep='\t', low_memory=False)
test_data = pd.read_csv(test_file, sep='\t', low_memory=False)

# Shuffle the datasets
train_data = shuffle(train_data)
test_data = shuffle(test_data)

# Save the shuffled data back to the files
train_data.to_csv(output_folder + 'train.log.txt', sep='\t', index=False)
test_data.to_csv(output_folder + 'test.log.txt', sep='\t', index=False)

print("Merging, shuffling, and saving completed.")
