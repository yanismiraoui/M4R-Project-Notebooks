import click
import pandas as pd

import vampire_common as common

translation_csv = 'adaptive-olga-translation.csv'


def adaptive_to_olga_dict():
    gb = common.read_data_csv(translation_csv).dropna().groupby('locus')
    return {locus: {row['adaptive']: row['olga'] for _, row in df.iterrows()} for locus, df in gb}


def olga_to_adaptive_dict():
    gb = common.read_data_csv(translation_csv).dropna().groupby('locus')
    return {locus: {row['olga']: row['adaptive'] for _, row in df.iterrows()} for locus, df in gb}


def filter_by_gene_names(df, conversion_dict):
    """
    Only allow through gene names that are present for both programs.
    """
    allowed = {locus: set(d.keys()) for locus, d in conversion_dict.items()}
    return df.loc[df['v_gene'].isin(allowed['TRBV']) & df['j_gene'].isin(allowed['TRBJ']), :]


def convert_gene_names(df, conversion_dict):
    converted = df.copy()
    converted['v_gene'] = df['v_gene'].map(conversion_dict['TRBV'].get)
    converted['j_gene'] = df['j_gene'].map(conversion_dict['TRBJ'].get)
    return converted


def convert_and_filter(df, conversion_dict):
    orig_len = len(df)
    converted = convert_gene_names(filter_by_gene_names(df, conversion_dict), conversion_dict)
    n_trimmed = orig_len - len(converted)
    if n_trimmed > 0:
        click.echo(f"Warning: couldn't convert {n_trimmed} sequences and trimmed them off.")
    return converted
