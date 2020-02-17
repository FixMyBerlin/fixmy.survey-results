from pathlib import Path
from os.path import join
import pandas as pd
import os
from dotenv import find_dotenv, load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[2]

def load_ratings():
    data_df = pd.read_csv(join(PROJECT_DIR, "data", "interim", "ratings.csv"), low_memory=False)
    return data_df


def load_profiles():
    data_df = pd.read_csv(join(PROJECT_DIR, "data", "interim", "profiles_df.csv"), low_memory=False).set_index("sessionID")
    return data_df


def load_full_data():
    data_df = pd.read_csv(join(PROJECT_DIR, "data", "processed", "full_df.csv"),
                          low_memory=False)
    return data_df
