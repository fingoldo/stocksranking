from stocksranking import *

from pyutilz.logginglib import init_logging

logger = init_logging(default_caller_name="stocksranking.py", format="%(asctime)s - %(levelname)s - %(funcName)s-line:%(lineno)d - %(message)s")

if __name__=="__main__":
    update_okx_hist_data()
    create_bulk_features(asset_class = "spot")
    featurized_files = read_feature_files(asset_class="spot")
    perf = train_models(featurized_files)
    perf.to_html('training_perf.html')