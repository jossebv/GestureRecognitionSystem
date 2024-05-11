import pandas as pd
import numpy as np
import os 
from tqdm import tqdm

# PARAMS
NUMB_INTERP = 2

#INTERPOLATION FUNCTION

# Function to insert row in the dataframe
def insert_row(row_number, df, row_value):
	# Starting value of upper half
	start_upper = 0
	# End value of upper half
	end_upper = row_number
	# Start value of lower half
	start_lower = row_number
	# End value of lower half
	end_lower = df.shape[0]
	# Create a list of upper_half index
	upper_half = [*range(start_upper, end_upper, 1)]
	# Create a list of lower_half index
	lower_half = [*range(start_lower, end_lower, 1)]
	# Increment the value of lower half by 1
	lower_half = [x.__add__(1) for x in lower_half]
	# Combine the two lists
	index_ = upper_half + lower_half
	# Update the index of the dataframe
	df.index = index_
	# Insert a row at the end
	df.loc[row_number] = row_value
	# Sort the index labels
	df = df.sort_index()
	# return the dataframe
	return df


def interpolate(df, frames_interpolated):
    new_df = df.copy()
    row_value = np.empty((len(new_df.columns)))
    row_value[:] = np.nan
    if (frames_interpolated > 0):
        ##INTERPOLATION OF 1 ROW##
        if (frames_interpolated == 1):
            for row_number in range(1,len(new_df)*2-1,2):
                new_df = insert_row(row_number, new_df, row_value)
        ##INTERPOLATION OF 2 ROWS##
        elif (frames_interpolated == 2):
            for row_number in range(1,len(new_df)*3-2,3):
                new_df = insert_row(row_number, new_df, row_value)
                new_df = insert_row(row_number, new_df, row_value)
        ##INTERPOLATION OF 3 ROWS##
        elif (frames_interpolated == 3):
            for row_number in range(1,len(new_df)*4-3,4):
                new_df = insert_row(row_number, new_df, row_value)
                new_df = insert_row(row_number, new_df, row_value)
                new_df = insert_row(row_number, new_df, row_value)
           
        #We interpolate
        for j, col in enumerate(new_df.columns):
            new_df[col] = new_df[col].interpolate()
    return new_df


print(f"Configuration: Number of frames interpolated = {NUMB_INTERP}.\nInterpolating landmarks...")
os.makedirs(f"../features/IPN_Hand/pose_features_interp_{NUMB_INTERP}/", exist_ok=True)
for file in tqdm(os.listdir("../features/IPN_Hand/pose_features_w_interp/")):
    landmarks_file = pd.read_csv(f"../features/IPN_Hand/pose_features_w_interp/{file}")
    landmarks_file_interp = interpolate(landmarks_file, NUMB_INTERP)
    os.makedirs(f"../features/IPN_Hand/pose_features_w_interp_interp_{NUMB_INTERP}/", exist_ok=True)
    landmarks_file_interp.to_csv(f"../features/IPN_Hand/pose_features_w_interp_interp_{NUMB_INTERP}/{file}", index=False)