#********************************************************************************************************************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

#********************************************************************************************************************************************************************************************************************************************************
# IMPORTS
#********************************************************************************************************************************************************************************************************************************************************

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# General
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

from typing import *

from server import app, APP_NAME, WEBSITE_NAME


import pytz
from tqdm import tqdm
from datetime import datetime
#from pyutilz.pythonlib import open_safe_shelve


from stocksranking import *
from os.path import join,getmtime

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DASH
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import dash_ag_grid as dag

#********************************************************************************************************************************************************************************************************************************************************
# INITS
#********************************************************************************************************************************************************************************************************************************************************

LAST_N_DAYS=2
TIMEZONE='CET'

our_statuses_style_conditions=[
       
        {
            "condition": "params.data.pred_close_to_open_rank > 0.15",
            "style": {"backgroundColor": "darkseagreen"},
        },
        {
            "condition": "params.data.pred_close_to_open_rank > 0.1",
            "style": {"backgroundColor": "sandybrown"},
        },      
                                               
    ]

our_rowstyles = {
    "styleConditions": our_statuses_style_conditions
}           

price_formatting={"function": "params.value ? d3.format(',.6f')(params.value) : '' "}
profit_formatting={"function": "params.value ? d3.format('.0%')(params.value) : '' "}
volume_formatting={"function": "params.value ? d3.format('.2f')(params.value) : '' "}
integer_formatting={"function": "params.value ? d3.format(',.0f')(params.value) : '' "}

price_width=140
dates_width=120
profit_width=170

highlighted_header_class="text-primary bg-warning"

grid_style={"height": '800px',} #  "width": "100%"
        
dashGridOptions={'pagination':True,'paginationPageSize':50,"maintainColumnOrder": True,"tooltipShowDelay": 200,"accentedSort": True,"columnHoverHighlight": True,                             
                    #"domLayout": "autoHeight"
                    "enableCellTextSelection": True, "ensureDomOrder": True,
                    'rowSelection': 'multiple',
                    #"rowHeight": 120,
                    }

defaultColDef={"sortable":True,"resizable":True,"filter":True,"suppressMovable":True,"wrapText": True,
                'cellStyle': {'textAlign': 'center','verticalAlign': 'middle'},
                "icons": {"sortAscending": '<i class="fa fa-sort-alpha-up" style="color: #66c2a5">',
                        "sortDescending": '<i class="fa fa-sort-alpha-down" style="color: #e78ac3"/>',},
}

columnDefs=[dict(
                    headerName="Info",
                    headerTooltip="Coin params",
                    openByDefault= True,            
                    children= [       
                        dict(field='ticker', headerName='Ticker',headerTooltip="Coin ticker",width= 200,cellRenderer="OKxLink")
                    ],
                    marryChildren=True
                ),
                
            dict(
                    headerName="Prev day",
                    headerTooltip="Previous trading day params",
                    openByDefault= True,            
                    children= [                                                       
                        dict(field= 'ntrades',headerName="NTrades",headerTooltip="Number of trades",valueFormatter=integer_formatting,filter="agNumberColumnFilter",width=price_width),
                        dict(field= 'total_btc_volume',headerName="Total volume",headerTooltip="Total BTC volume",
                             valueGetter= {"function": "Number(params.data.vol_mean)*Number(params.data.ntrades)"},
                             valueFormatter=volume_formatting,filter="agNumberColumnFilter",width=price_width),
                    ],
                    marryChildren=True
                ),
            dict(
                    headerName="Predictions",
                    headerTooltip="All kinds of predictions made for the next trading day",
                    openByDefault= True,            
                    children= [          	                                              
                        dict(field= 'pred_price_max',headerName="Max Return",headerTooltip="Predicted High/Open return",
                             valueGetter= {"function": "Math.exp(Number(params.data.pred_price_max))-1"},                             
                             valueFormatter=profit_formatting,filter="agNumberColumnFilter",width=profit_width,headerClass=highlighted_header_class),
                        dict(field= 'pred_close_to_open_rank',headerName="Top-25 prob",headerTooltip="Top-25 landing probability at the end of the trading day (Close/Open)",
                             valueFormatter=profit_formatting,sort='desc',filter="agNumberColumnFilter",width=profit_width,headerClass=highlighted_header_class),

                    ],
                    marryChildren=True
                )                
        ]

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

#********************************************************************************************************************************************************************************************************************************************************
# HELPERS
#********************************************************************************************************************************************************************************************************************************************************


 
 
#********************************************************************************************************************************************************************************************************************************************************
# USER INTERFACE
#********************************************************************************************************************************************************************************************************************************************************


def layout():

    #--------------------------------------------------------------------------------------------------------------
    # Main table
    #--------------------------------------------------------------------------------------------------------------

    update_okx_hist_data(last_n_days=LAST_N_DAYS)
    create_bulk_features(asset_class="spot",last_n_days=LAST_N_DAYS)

    features, last_fname, last_known_file=read_last_known_features_file(asset_class = "spot")    
    if features is not None:

        last_ingest_ts=getmtime(last_known_file)
        tz = pytz.timezone(TIMEZONE)
        last_ingest_ts=f'Created: {datetime.fromtimestamp(last_ingest_ts, tz).strftime("%d.%m %H:%M")}'        

        processed_features = features.drop(columns=DROP_COLUMNS)
        for target, estimator in tqdm(zip(targets, estimators), desc="model",):

            model_name = f"{estimator.__name__}_{target}"
            model = joblib.load(join("models", f"{model_name}.dump"))

            if type(model).__name__ == "CatBoostClassifier":
                preds = model.predict_proba(processed_features)[:, 1]
            else:
                preds = model.predict(processed_features)

            features["pred_" + target] = preds

        grid = dag.AgGrid(  
            id="coins",
            rowData=features.to_dict("records"),
            columnDefs=columnDefs,
            className="ag-theme-alpine compact",
            #className="ag-theme-alpine-dark",
            #columnSize="autoSize", # "autoSizeSkipHeader",
            #columnSizeOptions={"skipHeader": True,},
            dashGridOptions=dashGridOptions,
            defaultColDef=defaultColDef,
            getRowStyle=our_rowstyles,
            style=grid_style,
            persistence=True,
            persisted_props=["filterModel"]        
        )       
    else:
        grid=None
    
    
    #--------------------------------------------------------------------------------------------------------------
    # NavBar
    #--------------------------------------------------------------------------------------------------------------
    
    navbar = dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    # Use row and col to control vertical alignment of logo / brand
                    dbc.Row(
                        [
                            dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                            dbc.Col(dbc.NavbarBrand("CoinsRanker", className="ms-2")),
                        ],
                        align="center",
                        className="g-0",
                    ),
                    href="#",
                    style={"textDecoration": "none"},
                ),
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                dbc.Collapse(
                    [dbc.Badge("File: "+last_fname.replace('.zip',''), id="features_file", color="primary", className="me-1"),
                    dbc.Tooltip(
                        "Name of the last features file available.",
                        target="features_file",
                    ),                      
                    dbc.Badge(last_ingest_ts, id="last_ingest_ts",color="secondary", className="me-1"),
                    dbc.Tooltip(
                        "CET time when the features file was created & prediction was made.",
                        target="last_ingest_ts",
                    ),                     
                     ],
                    #html.Div(last_fname),
                    id="navbar-collapse",
                    is_open=False,
                    navbar=True,
                ),
            ]
        ),
        color="dark",
        dark=True,
    )        
    #--------------------------------------------------------------------------------------------------------------
    # Footer
    #--------------------------------------------------------------------------------------------------------------

    footer=html.Div(
                html.Div(f"© {datetime.now().year} {WEBSITE_NAME}",className="text-center")
    )    
    
    trigger=dcc.Interval(id="trigger", interval=10_000)  # проверяет, надо ли загружать обновленные файлы
    dummy=html.Div(id='dummy', style={'display':'none'})
    
    return dbc.Container([navbar,grid,dummy,footer],id='main-page-content',fluid=True)


# add callback for toggling the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open