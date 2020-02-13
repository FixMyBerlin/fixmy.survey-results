import pandas as pd
import numpy as np
from scipy.stats import kruskal
from scipy.stats import chi2
from sklearn.metrics import log_loss
import mord
from src.visualization.visualize import likert_plot_hypo
from src.features.build_features import onehot_df
import bootstrapped.bootstrap as bs
import bootstrapped.compare_functions as bs_compare
import bootstrapped.stats_functions as bs_stats
import bootstrapped.power as bs_power


remove = ["scene_id", "sessionID", "userGroup", "zipcode", "offended", "annoyingTraffic","berlinTraffic_0", "perspective", "responsible_0"]
other_drops = ["climateTraffic", "sharingConditions", "sharingModes", "saveSpace", "annoyingPeople", "introSelection", "Tr_re-Breite", "Tr_li-Breite", "RVA-Breite", "Nutzfläche"]
keep_float = ["index","ageGroup", "bicycleUse", "duration", "rating",
              "berlinTraffic_accidents", "berlinTraffic_aggression", "berlinTraffic_maintenance",
              "berlinTraffic_noise", "berlinTraffic_polution", "berlinTraffic_rules", "berlinTraffic_traffic",
             "motivationalFactors_safe", "motivationalFactors_faster", "motivationalFactors_bikefun",
              "motivationalFactors_weather", "transportRatings_car", "transportRatings_public",
              "transportRatings_bicycle", "transportRatings_motorbike", "transportRatings_pedestrian",
             "bikeReasons_infrastructure", "bikeReasons_8", "bikeReasons_distance", "bikeReasons_children",
             "bikeReasons_equipment", "bikeReasons_skills", "bikeReasons_physicalStrain", "bikeReasons_social",
             "vehiclesOwned_bicycle", "vehiclesOwned_carsharing", "vehiclesOwned_public", "vehiclesOwned_car",
              "vehiclesOwned_pedelec", "vehiclesOwned_motorbike", "hasChildren"]#, "perspective", "Kamera"]


def run_simulation(group1, group2):
    results = []
    for i in range(3000):
        results.append(bs.bootstrap_ab(group1.to_numpy(), group2.to_numpy(), bs_stats.sum, bs_compare.percent_change))    
    return results


def likelihood_ratio_test(features_alternate, labels, lr_model, features_null=None):
    """
    Based on: https://gist.github.com/rnowling/ec9c9038e492d55ffae2ae257aa4acd9
    Compute the likelihood ratio test for a model trained on the set of
    features in `features_alternate` vs a null model.  If `features_null`
    is not defined, then the null model simply uses the intercept (class
    probabilities).  Note that `features_null` must be a subset of
    `features_alternative` -- it can not contain features that are not
    in `features_alternate`. Returns the p-value, which can be used to
    accept or reject the null hypothesis.
    """
    labels_df = labels
    labels = np.array(labels)
    features_alternate = np.array(features_alternate)
    if features_null is not None:
        features_null = np.array(features_null)

        if features_null.shape[1] >= features_alternate.shape[1]:
            raise ValueError(
                "Alternate features must have more"
                "features than null features")

        lr_model.fit(features_null, labels)
        null_prob = lr_model.predict_proba(features_null)
        df = features_alternate.shape[1] - features_null.shape[1]
    else:
        null_prob = labels_df.value_counts()/labels_df.shape[0]
        null_prob = np.repeat(np.expand_dims(null_prob.to_numpy(), axis=1),
                              labels_df.shape[0], axis=1).T
        df = features_alternate.shape[1]

    lr_model.fit(features_alternate, labels)
    alt_prob = lr_model.predict_proba(features_alternate)

    alt_log_likelihood = -log_loss(labels,
                                   alt_prob,
                                   normalize=False)
    null_log_likelihood = -log_loss(labels,
                                    null_prob,
                                    normalize=False)

    G = 2 * (alt_log_likelihood - null_log_likelihood)
    p_value = chi2.sf(G, df)

    return p_value


def build_likelihood_ratio_test(df, column, name=""):
    y = df["rating"].astype("int32")
    for col in df:
        # get dtype for column
        dt = df[col].dtype
        # check if it is a number
        if dt == float:
            df[col].fillna(0, inplace=True)
        else:
            df[col].fillna("", inplace=True)
    df = df.loc[:, ~df.columns.isin(remove)]
    filter_drops = []
    for drop in other_drops:
        filter_drops = filter_drops + [x for x in df.columns if drop in x]
    df = df.drop(filter_drops, axis=1)
    keep_float_df = [x for x in keep_float if x in df.columns]
    df, _ = onehot_df(df, df, exclude=keep_float_df)
    df_null_drops = [x for x in df.columns if column in x] + ["rating"]
    df_null = df.drop(df_null_drops, axis=1)
    lr_test = likelihood_ratio_test(df.drop(["rating"], axis=1),
                                    y,
                                    mord.LogisticAT(),
                                    df_null)
    if lr_test < 0.05:
        result = "Signifikanter Unterschied"
    else:
        result = "Kein Signifikanter Unterschied"
    print("Ergebnis Likelihood Ratio Test:", result)


def test_hypothesis(group1, group2, name, group_names):
    group1 = group1.groupby(
            ['sessionID']).median()["rating"].rename(group_names[0])
    group2 = group2.groupby(
            ['sessionID']).median()["rating"].rename(group_names[1])
    likert_plot_hypo(group1, group2, name)
    print("Mittelwert von ", group1.name,
          "-", group1.mean())
    print("Mittelwert von ", group2.name,
          "-", group2.mean())
    bs_result = bs.bootstrap_ab(group1.to_numpy(),
                                group2.to_numpy(),
                                bs_stats.median,
                                bs_compare.percent_change)
    mean_change = bs_compare.percent_change(group1.mean(), group2.mean())
    print("Bootstrap Ergebnis:", bs_result)
    print("Unterschied im Mittelwert von",
          group1.name, "zu", group2.name, "(in Prozent)", mean_change)
    print("Ist der Unterschied signifikant?", bs_result.is_significant())


def kruskal_test(df, exclude=['duration', 'rating', 'sessionID']):
    for var in df.columns:
        if var in exclude:
            continue
        else:
            df = df.groupby(var)
            groups = [df.get_group(x)["rating"] for x in df.groups]
            print(var, kruskal(*groups))


# Function to calculate missing values by column
def missing_values_table(df):
    """Creates a table with an overview of all the porportions
    of the missing values of all columns

    Parameters:
    df (DataFrame): Data for the table

    Returns: None
    """
    # Total missing values
    mis_val = df.isnull().sum()
    
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    
    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    
    # Return the dataframe with missing information
    return mis_val_table_ren_columns