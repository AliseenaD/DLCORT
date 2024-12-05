import pandas as pd

# Read the data
df = pd.read_csv('510F_1_d2/510F_1_d2DLC_resnet50_ObjectTestNov21shuffle1_250000.csv')

# Get the 6th column (index 5) from row 3 onwards
probabilities = df.iloc[2:, 9].astype(float)  # 2: means start from 3rd row (index 2), 5 is the 6th column

# Calculate percentage above 0.9
percentage = (probabilities > 0.9).mean() * 100

print(f"Percentage of left ear probabilities above 0.9: {percentage:.2f}%")