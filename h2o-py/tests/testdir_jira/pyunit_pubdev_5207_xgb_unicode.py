import pandas as pd
from h2o.estimators.xgboost import *
from tests import pyunit_utils


def xgboost_unicode():
    assert H2OXGBoostEstimator.available()

    unic_df = pd.DataFrame({u'\xA5': [2, 3, 1], 'y': [0, 0, 1], 'x': [0.3, 0.1, 0.9]})
    h2o_unic = h2o.H2OFrame(unic_df, destination_frame="unic_df")

    xg1 =  H2OXGBoostEstimator(model_id = 'xg1', ntrees = 3)
    xg1.train(x = [u'\xA5', 'x'], y = "y", training_frame=h2o_unic)

if __name__ == "__main__":
    pyunit_utils.standalone_test(xgboost_unicode)
else:
    xgboost_unicode()
