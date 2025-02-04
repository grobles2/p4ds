import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from statsmodels.formula.api import ols


def make_zipcode_groups(data_set: pd.DataFrame) -> list[set]:
    """
    Groups together zipcodes with similar house prices
    :param data_set: it can be the training or test set
    :returns: he generated zipcode groups. Each zipcode group is a set
    """
    # firsr we perform a post-hoc t-test with correction for each zipcode pair
    pair_t = ols("price ~ zipcode", data_set).fit().t_test_pairwise("zipcode")
    # we retain all the zipcode pairs for which the post-hoc test fails to reject the null hypothesis
    # these are all the zipcode pairs whose average house price is not significantly different with a 95% CI
    t_tests_frame = pair_t.result_frame[pair_t.result_frame["reject-hs"].isin([False])].reset_index(names="zipcode-pairs")
    # we extract the zipcode pairs as a list
    zipcode_pairs = t_tests_frame["zipcode-pairs"].str.split("-").tolist()
    # we group together zipcodes with "similar" average house prices
    zipcode_groups = []
    for zipcode in data_set["zipcode"].unique():
        if any(zipcode in group for group in zipcode_groups):
            continue
        new_group = { zipcode }
        for pair in zipcode_pairs:
            if zipcode in pair:
                new_group.update(pair)
        zipcode_groups.append(new_group)
    return zipcode_groups


def assign_to_zipcode_group(zipcode: str, zipcode_groups: list[set], group_prefix="zg") -> str:
    """
    given a zipcode and a list of groups it assign a zipcode to a group
    The groups will be called using the group_prefix as a prefix
    """
    try:
        res = next(i for i, group in enumerate(zipcode_groups) if zipcode in group)
        return f"{group_prefix}_{res}"
    except StopIteration:
        # we need to take into account that some zipcode may be missing in the training set and present in the test set
        return f"{group_prefix}_other"
    

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, data, labels=None) -> "FeatureEngineeringTransformer":
        self.zipcode_groups = make_zipcode_groups(pd.concat([data["zipcode"], labels], axis=1))
        return self

    def transform(self, data, labels=None) -> pd.DataFrame:
        # let's make a copy of the original dataset
        engineered_data = data.copy()    
        # create a variable "north_loc", 1 for latitudes greather than 47.5
        engineered_data["north_loc"] = np.where(engineered_data["lat"] > 47.5, 1, 0)
        # create a variable "rural_east", 1 for longitudes
        engineered_data["rural_east"] = np.where(engineered_data["long"] > -121.6, 1, 0)
        # drop "lat" and "long"
        engineered_data = engineered_data.drop(columns=["lat", "long"])
        # drop "sqft_living15" and "sqft_living_above"
        engineered_data = engineered_data.drop(columns=["sqft_living15", "sqft_above"])
        # add a binary variable for the presence of a  basement
        engineered_data["basement"] = np.where(engineered_data["sqft_basement"] > 0.0, 1, 0)
        # drop "sqft_basement"
        engineered_data = engineered_data.drop(columns=["sqft_basement"])
        # create an "is_renovated" binary variable
        max_year = max(engineered_data["yr_built"].max(), engineered_data["yr_renovated"].max())
        engineered_data["renovated"] = np.where(
            (engineered_data["yr_built"] > max_year - 25) | (engineered_data["yr_renovated"] > 0), 1, 0
        )
        # drop "yr_built" and "yr_renovated"
        engineered_data = engineered_data.drop(columns=["yr_built", "yr_renovated"])
        # group zipcodes into groups (as we did last week)
        engineered_data["zipcode_group"] = engineered_data["zipcode"].apply(assign_to_zipcode_group, zipcode_groups=self.zipcode_groups)
        # drop "zipcode"
        engineered_data = engineered_data.drop(columns=["zipcode"])
        # replace 0 bathrooms with NaN (we will fill it later on)
        engineered_data.loc[engineered_data['bathrooms'] == 0, 'bathrooms'] = np.nan
        # replace 33 bedrooms with Nan (we will fill it later on)
        engineered_data.loc[engineered_data['bedrooms'] == 33, 'bedrooms'] = np.nan
        # drop id and date (as we won't use them)
        engineered_data = engineered_data.drop(columns=["id", "date"])
        return engineered_data


def get_preprocessing_pipeline(train_data: pd.DataFrame) -> Pipeline:
    cat_feats = ["zipcode_group"]
    binary_feats = ["waterfront", "north_loc", "rural_east", "basement", "renovated"]
    label_feat = "price"
    num_feats = [
        el for el in list(
            train_data.select_dtypes(include=[np.number])
        ) if el not in binary_feats or el != label_feat
    ]
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median").set_output(transform="pandas")),
        ('std_scaler', StandardScaler().set_output(transform="pandas"))
    ])
    column_transformer = ColumnTransformer(
        (
            ("numerical", num_pipeline, num_feats),
            ("categorical", OneHotEncoder(categories='auto', sparse_output=False).set_output(transform="pandas"), cat_feats),
        ),
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    return make_pipeline(FeatureEngineeringTransformer(), column_transformer)