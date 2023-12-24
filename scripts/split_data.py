import click
from pathlib import Path
import pandas as pd


@click.command()
@click.argument('tsv_path')
def main(tsv_path: str, train_size: int=337):
    ann_df = pd.read_csv(tsv_path, sep='\t')

    train_df = ann_df[:train_size]
    val_df = ann_df[train_size:]

    tsv_path_obj = Path(tsv_path)

    train_df.to_csv(tsv_path_obj.parent / 'train_df.csv', index=False)
    val_df.to_csv(tsv_path_obj.parent / 'valid_df.csv', index=False)


if __name__ == '__main__':
    main()
