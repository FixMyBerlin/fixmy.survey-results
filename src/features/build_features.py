from pathlib import Path
from os.path import join
from sklearn.preprocessing import minmax_scale, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]


def get_district_from_plz(profiles_df):
    plz2district = pd.read_csv(join(PROJECT_DIR,
                                    "data/interim/plz2district.csv"),
                               index_col=0).set_index("PLZ")
    for subject in profiles_df.index:
        current = profiles_df.loc[subject].copy()
        if not current["district"] == "":
            continue
        zipcode = int(current["zipcode"])
        result = plz2district[plz2district.index == zipcode]
        if result.shape[0] > 0:
            if result.shape[0] > 1:
                print("Double PLZ:", result)
                continue
            else:
                subject_district = result["Stadtteil"].values[0]
                current["district"] = subject_district
        else:
            current["district"] = " - "
        profiles_df.loc[subject] = current
    return profiles_df


def generate_plz2district():
    plz2ortsteil = pd.read_excel(join(PROJECT_DIR,
                                      "data/external/Bundesland Berlin.xlsx"))
    ortsteil2district = pd.read_csv(join(PROJECT_DIR,
                                         "data/external/ort2bezirk.csv"),
                                    header=None, index_col=0)
    for ortsteil in range(plz2ortsteil.shape[0]):
        ort = plz2ortsteil.iloc[ortsteil]["Stadtteil"]
        if ort in ortsteil2district.index:
            plz = plz2ortsteil.iloc[ortsteil]["PLZ"]
            district = ortsteil2district.loc[ort][1]
            plz2ortsteil.iloc[ortsteil] = pd.Series([plz, district],
                                                    index=["PLZ", "Stadtteil"])
        else:
            continue
    plz2ortsteil = plz2ortsteil.drop_duplicates(subset=["PLZ"])
    plz2ortsteil.to_csv(join(PROJECT_DIR,
                             "data/interim/plz2district.csv"))


def onehot_df(train, test, exclude=["duration", "index"]):
    exclude_train = train[exclude]
    exclude_test = test[exclude]
    train = train.drop(exclude, axis=1)
    test = test.drop(exclude, axis=1)
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc.fit(train)
    features = train.columns
    train = pd.DataFrame(enc.transform(train),
                         columns=enc.get_feature_names(features))
    test = pd.DataFrame(enc.transform(test),
                        columns=enc.get_feature_names(features))
    for ex in exclude:
        train[ex] = minmax_scale(exclude_train[ex].to_numpy())
        test[ex] = minmax_scale(exclude_test[ex].to_numpy())
    return train, test


remove = ["userGroup", "zipcode", "offended",
          "annoyingTraffic", "berlinTraffic_0", "perspective",
          "scene_id", "sessionID", "bikeReasonsVar"]

other_drops = ["climateTraffic", "sharingConditions", "sharingModes",
               "saveSpace", "annoyingPeople", "introSelection", "responsible"]

scale_feature = ["index", "ageGroup", "bicycleUse", "duration",
                 "berlinTraffic_accidents", "berlinTraffic_aggression",
                 "berlinTraffic_maintenance", "berlinTraffic_noise",
                 "berlinTraffic_polution", "berlinTraffic_rules",
                 "berlinTraffic_traffic", "motivationalFactors_safe",
                 "motivationalFactors_faster", "motivationalFactors_bikefun",
                 "motivationalFactors_weather", "transportRatings_car",
                 "transportRatings_public", "transportRatings_bicycle",
                 "transportRatings_motorbike", "transportRatings_pedestrian",
                 "bikeReasons_infrastructure", "bikeReasons_8",
                 "bikeReasons_distance", "bikeReasons_children",
                 "bikeReasons_equipment", "bikeReasons_skills",
                 "bikeReasons_physicalStrain", "bikeReasons_social",
                 "vehiclesOwned_bicycle", "vehiclesOwned_carsharing",
                 "vehiclesOwned_public", "vehiclesOwned_car",
                 "vehiclesOwned_pedelec", "vehiclesOwned_motorbike",
                 "hasChildren", "perspective", "Tr_li-Breite", "RVA-Breite"]


def feature_engineering(df):
    global scale_feature
    global remove
    global other_drops
    df_new = df.copy()
    remove_df = [x for x in remove if x in df_new.columns]
    df_new = df_new.drop(remove_df, axis=1)
    filter_drops = []
    for drop in other_drops:
        filter_drops = filter_drops + [x for x in df.columns if drop in x]
    df_new = df_new.drop(filter_drops, axis=1)
    index_replace = df_new[df_new["index"] > 10]["index"].value_counts().index
    index_replace = pd.Series(np.repeat(11, len(index_replace)),
                              index=index_replace)
    df_new["index"] = df_new["index"].replace(index_replace)
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    df_new = pd.DataFrame(imp.fit_transform(df_new), columns=df_new.columns)
    scale_feature_df = [x for x in scale_feature if x in df_new.columns]
    one_hot_cols = df_new[
                    df_new.columns[
                        ~df_new.columns.isin(
                                        scale_feature_df)]].drop(
                                                            ["rating"],
                                                            Saxis=1).columns
    ct = ColumnTransformer([
                            ("scale", StandardScaler(), scale_feature_df),
                            ("passthrough", "passthrough", ["rating"]),
                            ("one_hot",
                                OneHotEncoder(handle_unknown='ignore',
                                              sparse=False),
                                one_hot_cols)])
    df_new = ct.fit_transform(df_new)
    columns = [*scale_feature_df,
               "rating",
               *ct.named_transformers_[
                    'one_hot'].get_feature_names(one_hot_cols)]
    df_new = pd.DataFrame(df_new, columns=columns, dtype="float")
    return df_new
