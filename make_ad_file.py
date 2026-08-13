import pandas as pd

url = "https://raw.githubusercontent.com/mathworks/Chemistry-Deep-Learning-GCN-Mutagenicity-Classification/main/AMES-csv-Data/AMES_All_Data.csv"
df = pd.read_csv(url, header=None, names=["SMILES", "label"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
train_df = df.iloc[:int(0.8 * len(df))]          # same split as train_ames.py
train_df["SMILES"].to_csv("train_smiles.txt", index=False, header=False)
print(f"Wrote {len(train_df)} training SMILES to train_smiles.txt")