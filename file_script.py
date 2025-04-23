import pandas as pd

files = [
    '/Users/alidaeihagh/Desktop/DLCORT/1000_2_d2/1000_2_d2DLC_resnet50_ObjectTestNov21shuffle1_250000.csv',
    '/Users/alidaeihagh/Desktop/DLCORT/1000_1_d2/1000_1_d2DLC_resnet50_ObjectTestNov21shuffle1_250000.csv',
    '/Users/alidaeihagh/Desktop/DLCORT/524B_LR_d2/524B_LR_d2DLC_resnet50_ObjectTestNov21shuffle1_250000.csv',
    '/Users/alidaeihagh/Desktop/DLCORT/515M-2-D2/515M-2-D2DLC_resnet50_ObjectTestNov21shuffle1_250000.csv',
    '/Users/alidaeihagh/Desktop/DLCORT/515M-1-D2/515M-1-D2DLC_resnet50_ObjectTestNov21shuffle1_250000.csv',
    '/Users/alidaeihagh/Desktop/DLCORT/515M-0-D2/515M-0-D2DLC_resnet50_ObjectTestNov21shuffle1_250000.csv',
    '/Users/alidaeihagh/Desktop/DLCORT/510F_3_d2/510F_3_d2DLC_resnet50_ObjectTestNov21shuffle1_250000.csv',
    '/Users/alidaeihagh/Desktop/DLCORT/510F_2_d2/510F_2_d2DLC_resnet50_ObjectTestNov21shuffle1_250000.csv',
    '/Users/alidaeihagh/Desktop/DLCORT/510F_1_d2/510F_1_d2DLC_resnet50_ObjectTestNov21shuffle1_250000.csv',
    '/Users/alidaeihagh/Desktop/DLCORT/508F-2-D2/508F-2-D2DLC_resnet50_ObjectTestNov21shuffle1_250000.csv',
    '/Users/alidaeihagh/Desktop/DLCORT/508F-1-D2/508F-1-D2DLC_resnet50_ObjectTestNov21shuffle1_250000.csv',
    '/Users/alidaeihagh/Desktop/DLCORT/508F-0-D2/508F-0-D2DLC_resnet50_ObjectTestNov21shuffle1_250000.csv',
    '/Users/alidaeihagh/Desktop/DLCORT/507X_3_d2/507X_3_d2DLC_resnet50_ObjectTestNov21shuffle1_250000.csv',
    '/Users/alidaeihagh/Desktop/DLCORT/507X_2_d2/507X_2_d2DLC_resnet50_ObjectTestNov21shuffle1_250000.csv',
    '/Users/alidaeihagh/Desktop/DLCORT/507X_1_d2/507X_1_d2DLC_resnet50_ObjectTestNov21shuffle1_250000.csv'
]

for file in files:
    df = pd.read_csv(file, skiprows=3)
    title = file.split('/')[5]
    print(f"{title}: {len(df)} rows")
