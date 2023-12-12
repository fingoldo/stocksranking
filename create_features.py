from stocksranking import *

from pyutilz.logginglib import init_logging

logger = init_logging(default_caller_name="stocksranking.py", format="%(asctime)s - %(levelname)s - %(funcName)s-line:%(lineno)d - %(message)s")

if __name__=="__main__":
    create_bulk_features(asset_class = "spot")