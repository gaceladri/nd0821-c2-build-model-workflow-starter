#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging

import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def interquantile_rule(dataframe, args):
    """
    Remove outliers following the interquantile rule

    params:
        df: dataframe (multiple columns)

    returns:
        filtered dataframe
    """
    # An elegant way to remove outliers with the Interquartile Rule!
    # ref: https://stackoverflow.com/questions/35827863/remove-outliers-in-pandas-dataframe-using-percentiles
    quantile1 = dataframe.quantile(args.lower_interquantile)
    quantile3 = dataframe.quantile(args.higher_interquantile)
    iqr = quantile3 - quantile1

    return dataframe[~((dataframe < (quantile1 - 1.5 * iqr)) | (dataframe > (quantile3 + 1.5 * iqr))
        ).any(axis=1)]


def go(args):
    """
    Main function
    """

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download the artifact from WANDB
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    dataframe = pd.read_csv(artifact_local_path)
    logger.info('Data frame loaded')

    # We remove some outliers programmatically
    dataframe = interquantile_rule(dataframe, args)
    logger.info(
        'Removing outliers with interquantile rule, now shape: \
             %s min: %f max: %f', dataframe.shape, dataframe.price.min, dataframe.price.max)

    # Save the dataframe to csv
    dataframe.to_csv(args.output_artifact)
    logger.info('CSV saved to %s', args.output_artifact)

    logger.info('Uploading the artifact to WANDB %s', args.output_artifact)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )

    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)

    logger.info('Artifact Uploaded to WANDB %s', args.output_artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")
    parser.add_argument(
        "--input_artifact",
        type=str,
        help="input artifact - sample csv file",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="name for the output artifact (e.g cleaned data.csv)",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="type of the output (e.g clean_sample)",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="the description of the dataframe to be loaded in WANDB (eg. clean_dataset.csv)",
        required=True
    )

    parser.add_argument(
        "--lower_interquantile",
        type=float,
        help="the lower interquantile to consider and filter the dataframe for (e.g 0.15)",
        required=True
    )

    parser.add_argument(
        "--higher_interquantile",
        type=float,
        help="the higher interquantile to consider and filter the dataframe for (e.g 0.99)",
        required=True
    )

    args = parser.parse_args()

    go(args)
