import pandas as pd
from pathlib import Path
from os.path import join
from sklearn.preprocessing import scale, minmax_scale, OrdinalEncoder, OneHotEncoder

project_dir = Path(__file__).resolve().parents[2]


def get_district_from_plz(profiles_df):
    plz2district = pd.read_csv(join(project_dir, "data/interim/plz2district.csv"), index_col=0).set_index("PLZ")
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
            if zipcode < 15000 and zipcode > 10000:
                current["district"] = " - "
            else:
                current["district"] = "Outsider"
        profiles_df.loc[subject] = current
    return profiles_df


def generate_plz2district():
    plz2ortsteil = pd.read_excel(join(project_dir, "data/external/Bundesland Berlin.xlsx"))
    ortsteil2district = pd.read_csv(join(project_dir, "data/external/ort2bezirk.csv"), header=None, index_col=0)
    for ortsteil in range(plz2ortsteil.shape[0]):
        ort = plz2ortsteil.iloc[ortsteil]["Stadtteil"]
        if ort in ortsteil2district.index:
            plz2ortsteil.iloc[ortsteil] = pd.Series([plz2ortsteil.iloc[ortsteil]["PLZ"], ortsteil2district.loc[ort][1]], index=["PLZ", "Stadtteil"])
        else:
            continue
    plz2ortsteil = plz2ortsteil.drop_duplicates(subset=["PLZ"])
    plz2ortsteil.to_csv(join(project_dir, "data/interim/plz2district.csv"))


def onehot_df(train, test, exclude=["duration", "index"]):
    exclude_train = train[exclude]
    exclude_test = test[exclude]
    train = train.drop(exclude, axis=1)
    test = test.drop(exclude, axis=1)
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc.fit(train)
    features = train.columns
    train = pd.DataFrame(enc.transform(train), columns=enc.get_feature_names(features))
    test = pd.DataFrame(enc.transform(test), columns=enc.get_feature_names(features))
    for ex in exclude:
        train[ex] = minmax_scale(exclude_train[ex].to_numpy())
        test[ex] = minmax_scale(exclude_test[ex].to_numpy())
    return train, test
