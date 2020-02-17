# -*- coding: utf-8 -*-
"""Transforms our external and raw data into our main dataframe"""
from pathlib import Path
from os.path import join
import logging
import json
import numpy as np
import pandas as pd

import src
from src.features import build_features

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = join(PROJECT_DIR, "data", "raw")
INTERIM_DIR = join(PROJECT_DIR, "data", "interim")
PROCESSED_DIR = join(PROJECT_DIR, "data", "processed")

drop_list = ["perspective"]


def preprocessing(filename):
    """Load raw data file and create dataframes (csv files)
       for the ratings and the profiles

    Arguments:
        filename {str} -- filename of the raw data used
    """
    logger = logging.getLogger(__name__)
    data = json.load(open(join(RAW_DIR, filename), "rb"))
    profiles_df = pd.DataFrame()
    data_df = pd.DataFrame()
    logger.info('raw data contains %s entries', len(data))
    for data_point in data:
        rating_df = pd.DataFrame(data_point["ratings"])
        rating_df["sessionID"] = data_point["profile"]["sessionID"]
        rating_df = rating_df.reset_index(level=0)
        data_df = data_df.append(rating_df, ignore_index=True, sort=False)
        profiles_df = profiles_df.append(pd.Series(data_point["profile"]),
                                         ignore_index=True)
    profiles_df = profiles_df.set_index("sessionID")
    exclude = [x for x in ["berlinTraffic_0",
                           "isTosAccepted"] if x in profiles_df.columns]
    profiles_df = profiles_df.drop(exclude, axis=1)
    dict_columns = [x for x in src.dict_columns if x in profiles_df.columns]
    list_columns = [x for x in src.list_columns if x in profiles_df.columns]
    for col in dict_columns:
        new_columns = profiles_df[col].apply(pd.Series)
        new_columns.columns = [col + "_" + str(name)
                               for name in new_columns.columns]
        profiles_df = pd.concat([profiles_df.drop([col], axis=1),
                                 new_columns],
                                axis=1)
    for col in list_columns:
        for subject in profiles_df.index:
            answers = profiles_df.loc[subject][col]
            counts = pd.Series(
                     answers,
                     dtype="object").value_counts().rename(
                                                    subject).add_prefix(
                                                             col + "_")
            for i in counts.index:
                if i not in profiles_df.columns:
                    profiles_df[i] = np.nan
            copy = profiles_df.loc[subject].copy()
            copy[counts.index] = counts
            profiles_df.loc[subject] = copy
        profiles_df = profiles_df.drop(col, axis=1)
    profiles_df = build_features.get_district_from_plz(profiles_df)
    profiles_df.to_csv(join(INTERIM_DIR, "profiles_df.csv"))
    data_df.to_csv(join(INTERIM_DIR, "ratings.csv"), index=False)


def merging():
    """Merge subject profiles and ratings to a single dataframe with
       the parameters of the scenes
    """
    profiles_df = pd.read_csv(join(INTERIM_DIR, "profiles_df.csv"),
                              low_memory=False)
    data_df = pd.read_csv(join(INTERIM_DIR, "ratings.csv"), low_memory=False)
    rename = {"SceneID": "scene_id"}
    scenes_cp = pd.read_csv(
                join(RAW_DIR,
                     "scenes_cp.csv")).rename(columns=rename).drop(["weight",
                                                                    "Raus",
                                                                    "SR_lD"],
                                                                   axis=1)
    scenes_ms = pd.read_csv(
                join(RAW_DIR,
                     "scenes_ms.csv")).rename(columns=rename).drop(["weight",
                                                                    "HVS_lD",
                                                                    "Raus"],
                                                                   axis=1)
    scenes_se = pd.read_csv(
                join(RAW_DIR,
                     "scenes_se.csv")).rename(columns=rename).drop(["weight",
                                                                    "NVS_lD",
                                                                    "raus"],
                                                                   axis=1)
    full_df = pd.merge(data_df, profiles_df, how="left", on="sessionID")
    cp_full = pd.merge(full_df, scenes_cp, on="scene_id")
    ms_full = pd.merge(full_df, scenes_ms, on="scene_id")
    se_full = pd.merge(full_df, scenes_se, on="scene_id")
    full_df = pd.concat([cp_full, ms_full, se_full], sort=False)
    full_df.to_csv(join(PROCESSED_DIR, "full_df.csv"), index=False)


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    filename = input("Welche Rohdaten sollen geladen werden?" +
                     "(Dateiname aus raw Ordner):")
    logger.info('preprocessing data')
    preprocessing(filename)
    logger.info('merging data')
    merging()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    build_features.generate_plz2district()
    main()
