import pandas as pd
import torch

from torch.utils.data import Dataset


class ToxicDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ref = self.dataframe.iloc[idx, 1]
        trn = self.dataframe.iloc[idx, 2]

        return ref, trn


if __name__ == "__main__":
    print("Starting dataset processing... It may take some minutes, be partient.")

    filtered_dataset = pd.read_csv("data/raw/filtered.tsv", sep="\t")
    filtered_dataset.rename(columns={filtered_dataset.columns[0]: "id"}, inplace=True)

    condition = filtered_dataset["ref_tox"] < filtered_dataset["trn_tox"]

    filtered_dataset.loc[condition, ["reference", "translation"]] = (
        filtered_dataset.loc[condition, ["translation", "reference"]].values
    )

    filtered_dataset.loc[condition, ["ref_tox", "trn_tox"]] = (
        filtered_dataset.loc[condition, ["trn_tox", "ref_tox"]].values
    )

    torch_dataset = ToxicDataset(filtered_dataset)

    with open("data/interim/train_text.txt", "w") as out_file:
        for idx in range(len(torch_dataset)):
            out_file.write(
                f"User: {torch_dataset[idx][0]}\nAssistant: {torch_dataset[idx][1]}\n\n"
            )

    print("Processing completed. Data saved to `data/interim/train_text.txt`")
