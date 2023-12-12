    # ****************************************************************************************************************************
# Imports
# ****************************************************************************************************************************


# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import pandas as pd, numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from os.path import exists, join
import joblib
from datetime import date, timedelta

import urllib.request
import concurrent.futures
from urllib.request import HTTPError
# from pyutilz.pandaslib import optimize_dtypes

#from mlframe.feature_engineering.timeseries import create_aggregated_features
#from mlframe.feature_engineering.numerical import compute_numaggs, get_numaggs_names
#from mlframe.feature_engineering.numerical import compute_simple_stats_numba, get_simple_stats_names

from catboost import CatBoostRanker, CatBoostClassifier, CatBoostRegressor, Pool
from catboost.utils import eval_metric
import numba

from sklearn.dummy import DummyRegressor, DummyClassifier

# ****************************************************************************************************************************
# Inits
# ****************************************************************************************************************************

opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)

plt.rcParams["figure.figsize"] = (20, 10)

pd.set_option("display.precision", 5)
pd.set_option("display.max_rows", 50)
pd.set_option("display.max_columns", 50)

fastmath=True

#DATAPATH = r"R:\Data\StocksRanking"
DATAPATH = "data"

FIRST_HIST_DATE=date(2021, 10, 1)
DROP_COLUMNS = ["ticker"]

GENERAL_PARAMS = dict(iterations=10_000, task_type="CPU", early_stopping_rounds=1000)
targets, estimators, thresholds, metrics = (
    "close_to_open_rank price_max".split(),  # price_max
    [CatBoostClassifier, CatBoostRegressor],
    [0.95, None],
    ["AUC", "RMSE"],
)


# ****************************************************************************************************************************
# Features
# ****************************************************************************************************************************


@numba.njit(fastmath=fastmath)
def compute_simple_stats_numba(arr: np.ndarray)->tuple:
    minval,maxval,argmin,argmax=arr[0],arr[0],0,0
    size=len(arr)
    total,std_val=0.0,0.0

    for i,next_value in enumerate(arr):
        total+=next_value
        if next_value<minval:
            minval=next_value
            argmin=i
        elif next_value>maxval:
            maxval=next_value
            argmax=i
    mean_value=total/size

    for i,next_value in enumerate(arr):
        d = next_value - mean_value
        summand = d * d
        std_val = std_val + summand
    std_val = np.sqrt(std_val / size)
    return minval,maxval,argmin,argmax,mean_value,std_val

def get_simple_stats_names()->list:
    return "min,max,argmin,argmax,mean,std".split(",")

feature_names = (
    "ticker close_to_open mean_trade_direction ntrades market_volume_share market_mean_trade_direction".split()
    + ["vol_" + el for el in get_simple_stats_names()]
    + ["price_" + el for el in get_simple_stats_names()]
)

# ****************************************************************************************************************************
# Scraping
# ****************************************************************************************************************************

def update_okx_hist_data(last_n_days:int=None)->int:
    n=0
    
    if last_n_days:
        first_date= date.today() - timedelta(days=last_n_days)
    else:
        first_date=FIRST_HIST_DATE
    
    for dt in tqdm(pd.date_range(first_date,date.today() + timedelta(days=2))):
        for asset_class in "spot future swap".split():
            fname = f"all{asset_class}-aggtrades-{dt.strftime('%Y-%m-%d.zip')}"
            fpath = join(DATAPATH, fname)
            if not exists(fpath):
                url = f"https://www.okx.com/cdn/okex/traderecords/aggtrades/monthly/{dt.strftime('%Y%m')}/{fname}"
                try:
                    urllib.request.urlretrieve(
                        url, fpath,
                    )
                except HTTPError as e:
                    if e.code!=404:
                        logger.error(f"Error reading file {url}: {e}")
                else:
                    logger.info(f"Downloaded raw data file {fname}.")
                    n+=1
    return n

def read_okx_daily_trades(fname: str, clean: bool = False) -> pd.DataFrame:
    df = pd.read_csv(join(DATAPATH, fname), encoding="GBK", header=None, low_memory=True)
    assert df.shape[1] == 6
    if "instrument_name" in df.iat[0, 0]:
        df = df.iloc[1:, :]
    df.columns = "ticker tradeid side volume price ts".split()
    if clean:
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.sort_values(["ts", "tradeid"])
        try:
            df["side"] = df["side"].map({"buy": 1, "sell": -1, "BUY": 1, "SELL": -1}).astype(np.int8)
        except Exception as e:
            logger.error(f"File {fname} contains errors in Side colum. Add it to exclusions list.")
            return
        else:
            # optimize_dtypes(df, ensure_float64_precision=False, verbose=False, inplace=True)
            pass
    return df

# ****************************************************************************************************************************
# Preprocessing
# ****************************************************************************************************************************

def get_conversion_rates(df: pd.DataFrame, MAIN_CURRENCY: str = "BTC", verbose: bool = False) -> dict:

    first_prices = df.groupby("ticker").first().price.to_dict()
    vol_conversion_rates = {}
    rates = {}

    # direct
    for pair, price in first_prices.items():
        if "PERFTEST" in pair:
            continue
        coin1, coin2 = pair.split("-")
        if coin2 == MAIN_CURRENCY:
            vol_conversion_rates[pair] = price
            rates[coin1] = price
        elif coin1 == MAIN_CURRENCY:
            vol_conversion_rates[pair] = 1.0
            rates[coin2] = 1 / price

    # seconds pass
    for pair, price in first_prices.items():
        if "PERFTEST" in pair:
            continue
        coin1, coin2 = pair.split("-")
        if coin2 != MAIN_CURRENCY and coin1 != MAIN_CURRENCY:
            if coin2 in rates:
                vol_conversion_rates[pair] = rates.get(coin2) * price
            else:
                if verbose:
                    logger.info(f"Direct price for pair {pair} not found!")
                # try finding conversion with any of already known coins
                # we have transactions for pair BSB-RVN. but there is no BTC-RVN pair. But there are RVN-USDT and BTC-USDT pairs
                for cross_currency in rates.keys():
                    art_pair = f"{coin2}-{cross_currency}"
                    if art_pair in first_prices:
                        if verbose:
                            logger.info(f"Found price for {pair} through {art_pair}")
                        vol_conversion_rates[pair] = price * first_prices[art_pair] * rates[cross_currency]
                        break
                else:
                    if coin1 in rates:
                        vol_conversion_rates[pair] = rates[coin1]
                        if verbose:
                            logger.info(f"Found price for {pair} through {coin1}")
                    else:
                        logger.warning(f"Any price for {coin2} of pair {pair} not found!")
    return vol_conversion_rates

# ****************************************************************************************************************************
# ML
# ****************************************************************************************************************************

def create_interval_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    data = []
    iterable = df.ticker.unique()
    if verbose:
        iterable = tqdm(iterable)

    period_total_volume = df.unified_volume.sum()
    period_mean_trade_direction = df.side.mean()

    for ticker in iterable:
        if "PERFTEST" in ticker:
            continue

        idx = df.ticker == ticker

        prices = df[idx].price.values
        first_price, last_price = prices[0], prices[-1]
        # last_prices[ticker] = last_price

        close_to_open = np.log(last_price / first_price)
        mean_trade_direction = df[idx].side.mean()

        volumes = df[idx].unified_volume.values
        market_volume_share = volumes.sum() / period_total_volume

        ntrades = len(volumes)

        line = [ticker, close_to_open, mean_trade_direction, ntrades, market_volume_share, period_mean_trade_direction]
        minval, maxval, argmin, argmax, mean_value, std_val = compute_simple_stats_numba(volumes)
        line.extend((minval, maxval, argmin / ntrades, argmax / ntrades, mean_value, std_val))

        minval, maxval, argmin, argmax, mean_value, std_val = compute_simple_stats_numba(prices)
        # max_prices[ticker]=maxval
        minval = np.log(minval / first_price)
        maxval = np.log(maxval / first_price)
        mean_value = np.log(mean_value / first_price)
        std_val = np.log(std_val / first_price)

        line.extend((minval, maxval, argmin / ntrades, argmax / ntrades, mean_value, std_val))

        data.append(line)

    interval_features = pd.DataFrame(data, columns=feature_names)
    interval_features = interval_features[interval_features.ntrades > 1]

    # whole market features

    interval_features["close_to_open_rank"] = interval_features.close_to_open.rank(pct=True, ascending=True)
    interval_features["max_to_open_rank"] = interval_features.price_max.rank(pct=True, ascending=True)
    interval_features["market_volume_share_rank"] = interval_features.market_volume_share.rank(pct=True, ascending=True)
    interval_features["mean_trade_direction_rank"] = interval_features.mean_trade_direction.rank(pct=True, ascending=True)

    return interval_features

def featurize_raw_file(fname: str):
    df = read_okx_daily_trades(fname, clean=True)
    if df is not None:
        vol_conversion_rates = get_conversion_rates(df)
        df["unified_volume"] = df["ticker"].map(vol_conversion_rates) * df["volume"]
        df=df[~df.unified_volume.isna()]
        interval_features = create_interval_features(df)
        # optimize_dtypes(interval_features, ensure_float64_precision=False, verbose=False, inplace=True)
        interval_features.to_parquet(join(DATAPATH,'features',fname.replace('.zip','.parquet')))
        logger.info(f"Created features file {fname}.")

def get_indices(master, search):

    if not set(search).issubset(set(master)):
        raise ValueError("search must be a subset of master")

    sorti = np.argsort(master)

    # get indices in sorted version
    tmpind = np.searchsorted(master, search, sorter=sorti)

    final_inds = sorti[tmpind]

    return final_inds

def show_fi(model, X_test: pd.DataFrame, drop_columns: list = [], max_nfeatures: int = 30, title: str = None, figsize=(12, 6)):
    try:
        feature_importance = model.feature_importances_
    except:
        pass
    else:
        sorted_idx = np.argsort(feature_importance)
        fig = plt.figure(figsize=figsize)
        if max_nfeatures:
            sorted_idx = sorted_idx[:max_nfeatures]
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align="center")
        plt.yticks(range(len(sorted_idx)), np.array(X_test.columns if not drop_columns else X_test.head().drop(columns=drop_columns).columns)[sorted_idx])
        if title:
            title = "Feature Importance of " + title
        else:
            title = "Feature Importance"
        plt.title(title)
        plt.tight_layout()
        plt.show()

def create_bulk_features(asset_class:str = "spot",last_n_days:int=None):
    bad_files="allspot-aggtrades-2022-06-29.zip allspot-aggtrades-2022-07-21.zip allspot-aggtrades-2022-07-22.zip allspot-aggtrades-2022-07-23.zip allspot-aggtrades-2022-07-24.zip allspot-aggtrades-2022-07-25.zip".split()
    trading_days = []
    files = []

    if last_n_days:
        first_date= date.today() - timedelta(days=last_n_days)
    else:
        first_date=FIRST_HIST_DATE

    for dt in tqdm(pd.date_range(first_date, date.today() + timedelta(days=2))):
        # for asset_class in "spot future swap".split():
        tday = dt.strftime("%Y-%m-%d")
        fname = f"all{asset_class}-aggtrades-{dt.strftime('%Y-%m-%d.zip')}"
        if fname not in bad_files:
            fpath = join(DATAPATH, fname)
            if exists(fpath) and not exists(join(DATAPATH,'features',fname.replace('.zip','.parquet'))):
                trading_days.append(tday)
                files.append(fname)
    if files:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            featurized_files = [future.result() for future in concurrent.futures.as_completed([executor.submit(featurize_raw_file, file) for file in files])]

        return len(featurized_files)

def read_feature_files(asset_class: str = "spot") -> list:

    trading_days = []
    files = []
    for dt in tqdm(pd.date_range(date(2021, 10, 1), date.today() + timedelta(days=2))):
        # for asset_class in "spot future swap".split():
        tday = dt.strftime("%Y-%m-%d")
        fname = f"all{asset_class}-aggtrades-{dt.strftime('%Y-%m-%d.parquet')}"
        fpath = join(DATAPATH, "features", fname)
        if exists(fpath):
            trading_days.append(tday)
            files.append(fname)

    featurized_files = []
    if files:

        with concurrent.futures.ThreadPoolExecutor() as executor:
            featurized_files = [
                future.result()
                for future in concurrent.futures.as_completed([executor.submit(pd.read_parquet, join(DATAPATH, "features", file)) for file in files])
            ]

    return featurized_files

def create_ml_datasets(featurized_files: list, target_column: str = "max_to_open_rank", val_size: int = 10, test_size: int = 1,min_threshold:float=None,logarithmize:bool=False) -> tuple:
    # close_to_open_rank
    prev_df = None
    X_train, Y_train, groups_train = [], [], []
    X_val, Y_val, groups_val = [], [], []
    X_test, Y_test, groups_test = [], [], []
    i = 0
    
    for df in tqdm(featurized_files):
        if prev_df is not None:
            common_tickers = list(set(prev_df.ticker.values).intersection(set(df.ticker.values)))
            if common_tickers:
                # np.random.shuffle(common_tickers)

                new_features = prev_df.iloc[get_indices(master=prev_df.ticker.values, search=common_tickers)].drop(columns=DROP_COLUMNS)
                new_targets = df.iloc[get_indices(master=df.ticker.values, search=common_tickers)][target_column]
                if min_threshold:
                    new_targets=(new_targets>=min_threshold).astype(np.int8)
                if logarithmize:
                    new_targets=np.log1p(new_targets)

                if i < len(featurized_files) - (val_size + test_size + 1):
                    groups_train.extend([i] * len(common_tickers))
                    X_train.append(new_features)
                    Y_train.append(new_targets)

                elif i < len(featurized_files) - test_size - 1:
                    groups_val.extend([i] * len(common_tickers))
                    X_val.append(new_features)
                    Y_val.append(new_targets)
                else:
                    groups_test.extend([i] * len(common_tickers))
                    X_test.append(new_features)
                    Y_test.append(new_targets)

                i += 1
        prev_df = df

    train_pool = Pool(data=pd.concat(X_train, ignore_index=True), label=pd.concat(Y_train, ignore_index=True), group_id=groups_train)

    eval_pool = Pool(data=pd.concat(X_val, ignore_index=True), label=pd.concat(Y_val, ignore_index=True), group_id=groups_val)

    test_pool = Pool(data=pd.concat(X_test, ignore_index=True), label=pd.concat(Y_test, ignore_index=True), group_id=groups_test)

    return train_pool, eval_pool, test_pool, prev_df

def read_last_known_features_file(asset_class: str = "spot"):
    
    # find last features file & get predictions for it.

    last_known_file = None
    last_fname = None
    features = None

    for dt in tqdm(pd.date_range(FIRST_HIST_DATE, date.today() + timedelta(days=2))):

        fname = f"all{asset_class}-aggtrades-{dt.strftime('%Y-%m-%d.zip')}"
        fpath = join(DATAPATH, "features", fname.replace(".zip", ".parquet"))
        if exists(fpath):
            last_fname = fname
            last_known_file = fpath

    if last_known_file:
        features = pd.read_parquet(last_known_file)

    return features, last_fname, last_known_file

def train_models(featurized_files:Sequence,val_size:int = 5,test_size:int = 1)->pd.DataFrame:
    perf = []

    for target, estimator, min_threshold, metric, logarithmize in tqdm(zip(targets, estimators, thresholds, metrics, [False, False]), desc="model",):
        train_pool, eval_pool, test_pool, last_df = create_ml_datasets(
            featurized_files=featurized_files,
            target_column=target,
            val_size=val_size,
            test_size=test_size,
            min_threshold=min_threshold,
            logarithmize=logarithmize,
        )

        # --------------------------------------------------------------------------------------------------------------
        # Train
        # --------------------------------------------------------------------------------------------------------------

        params = GENERAL_PARAMS.copy()
        if estimator.__name__ == "CatBoostClassifier":
            params["eval_metric"] = "AUC"
        model = estimator(**params)

        model.fit(
            train_pool, eval_set=eval_pool, plot=True, verbose=False,
        )

        # show_fi(model, drop_columns=[], max_nfeatures=20, X_test=X_val[0])

        # --------------------------------------------------------------------------------------------------------------
        # Persist
        # --------------------------------------------------------------------------------------------------------------

        model_name = f"{estimator.__name__}_{target}"
        joblib.dump(model, join("models", f"{model_name}.dump"))

        # --------------------------------------------------------------------------------------------------------------
        # Evaluate & compare to baseline
        # --------------------------------------------------------------------------------------------------------------

        for data_set, data_set_name in zip([eval_pool, test_pool], ["val", "test"]):

            if estimator.__name__ == "CatBoostClassifier":
                preds = model.predict_proba(data_set)[:, 1]
                dummy_estimator = DummyClassifier
                strategies = "prior most_frequent stratified uniform".split()
            else:
                preds = model.predict(data_set)
                dummy_estimator = DummyRegressor
                strategies = "mean median".split()

            perf.append([target, estimator.__name__, data_set_name, eval_metric(data_set.get_label(), preds, metric)[0]])
            if True:
                for strategy in strategies:
                    dummy_model = dummy_estimator(strategy=strategy)
                    dummy_model.fit(train_pool, train_pool.get_label())
                    if estimator.__name__ == "CatBoostClassifier":
                        preds = dummy_model.predict_proba(data_set)[:, 1]
                    else:
                        preds = dummy_model.predict(data_set)
                    perf.append([target, dummy_estimator.__name__ + "_" + strategy, data_set_name, eval_metric(data_set.get_label(), preds, metric)[0]])

    perf = pd.DataFrame(perf, columns="target model fold metric".split())
    return perf
