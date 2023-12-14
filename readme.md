# CoinsRanker

## Settings

You might want to adjust some control parameters in stocksranking.py, for instance, data dir:

DATAPATH = r"R:\Data\StocksRanking"

## Install

You would require python 3.6+

```bash
pip install -r requirements.txt
```

## Retrain models

```bash
python retrain_models.py
```


## Start the dashboard locally

IN test mode, use

```bash
python app.py
```

or

```bash
python3 app.py
```

For production mode, use gunicorn, nginx or similar from Dash docs.

## Start the dashboard in cloud

in the last row of app.py, edit the port and host as needed. Example: port=80, host='your_public_ip'

Ensure that port is not used by something else and not firewalled on your server.