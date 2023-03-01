"""
Dash app entry point
To launch the app, run
> python app.py
Dash documentation: https://dash.plot.ly/
"""
import os
import glob
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px

import dash
from dash.dependencies import Input, Output, State, ClientsideFunction
from dash import dash_table
from dash import html
from dash import dcc

LABEL_FONT_SIZE = 20

FIRST_LINE_HEIGHT = 600

if 'DEBUG' in os.environ:
    debug = os.environ['DEBUG'] == 'True'
    print(f"DEBUG environment variable present, DEBUG set to {debug}")
else:
    print("No DEBUG environment variable: defaulting to debug mode")
    debug = True

def make_plot(list_of_csv):
    """
    Build figure showing evolution of number of cases vs. time for all countries.
    The visibility of traces is set to 0 so that the interactive app will
    toggle the visibility.
    Parameters
    ----------
    df_measure: pandas DataFrame
        DataFrame of measured cases, created by :func:`data_input.get_data`, of wide format.
    df_prediction: pandas DataFrame
        DataFrame of predictions, with similar structure as df_measure
    countries: list or None (default)
        list of countries to use for the figure. If None, all countries are used.
    """
    # Plot per million
    colors = px.colors.qualitative.Dark24
    n_colors = len(colors)
    fig = go.Figure()
    hovertemplate_prediction = '<b>%{meta}</b><br>x=%{x}<br>y=%{y}<extra></extra>'

    for index, path in enumerate(list_of_csv):
        df_temporary = pd.read_csv(path)
        print(path)
        print(df_temporary.head())
        print(df_temporary["timestamp"])
        quit()
        fig.add_trace(go.Scatter(x=df_temporary["timestamp"] / 1000,
                                 y=df_temporary["Relative optimality gap"], mode='lines',
                                 line_dash='dash',
                                 line_color=colors[index%n_colors],
                                 showlegend=False,
                                 meta=path.split(os.path.sep)[-1].rsplit(".", 1)[0],
                                 hovertemplate=hovertemplate_prediction,
                                 visible=True))

    fig.update_layout(
        showlegend=True,
        xaxis_tickfont_size=LABEL_FONT_SIZE - 4,
        yaxis_tickfont_size=LABEL_FONT_SIZE - 4,
        yaxis_type='linear',
        yaxis_title="Relative optimality gap",
        xaxis_title="Time (s)",
        hovermode="closest",
        height=FIRST_LINE_HEIGHT,
        margin=dict(t=0, b=0.02),
        # The legend position + font size
        # See https://plot.ly/python/legend/#style-legend
        legend=dict(x=.05, y=.8, font_size=LABEL_FONT_SIZE),
        yaxis=dict(range=[-6, 3]),
        width=1500
    )

    return fig

# -------- Data --------------------------

# ---------------- Figures -------------------------
fig2 = make_plot(glob.glob("./csvs/*.csv"))

# -----------App definition-----------------------
app = dash.Dash(__name__,
    external_stylesheets = [
        {
            'href': 'https://unpkg.com/purecss@1.0.1/build/pure-min.css',
            'rel': 'stylesheet',
            'integrity': 'sha384-oAOxQR6DkCoMliIh8yFnu25d7Eq/PHS21PClpwjOTeU2jRSq11vu66rf90/cZr47',
            'crossorigin': 'anonymous'
        },
        'https://unpkg.com/purecss@1.0.1/build/grids-responsive-min.css',
        'https://unpkg.com/purecss@1.0.1/build/base-min.css',
    ],
)
app.title = 'Cyanure: results on different datasets and with different processing conditions'
server = app.server
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

app.layout = html.Div([
    html.H1(children=app.title, className="title"),
    html.Div([#row
        html.Div([
            dcc.RadioItems(id='log-lin',
                options=[
                    {'label':'log', 'value': 'log'},
                    {'label': 'linear', 'value': 'linear'},
                ],
                value='linear',
                labelStyle={'display': 'inline-block',
                            'paddingRight': '0.5em'},
                style={'marginLeft': 125},
          ),

            dcc.Graph(
                id='plot', figure=fig2,
                config={
                    'displayModeBar': True,
                    'modeBarButtonsToRemove': ['toImage', 'zoom2d',
                                               'select2d', 'lasso2d',
                                               'toggleSpikelines',
                                               'resetScale2d']}
                , style={'marginLeft': 50},),
             dcc.Textarea(
                    value='The name of the curves correspond to nb_cores|implementation|penalty|lambda|duality gap interval|number of threads|solver|tolerance|dataset',
                    disabled=True,
                    style={'width': '100%', 'height': 50, 'marginLeft': 125, 'marginTop': 50},
                ),
            ]
            ),
        dcc.Store(id='store', data=fig2),
        html.Div(children=[
        html.Label('Datasets'),
        dcc.Dropdown(['real-sim', 'epsilon', 'rcv1', 'covtype', 'alpha', 'ckn_mnist', 'all'], 'ckn_mnist', id='dataset'),

        html.Br(),
        html.Label('Cores'),
        dcc.Dropdown(['1', '2', '4', '8', '16', '32', 'all'], '16', id='core'),

        html.Br(),
        html.Label('Implementation'),
        dcc.Dropdown(['openblas', 'mkl', 'blis', 'netlib', 'all'], 'mkl', id='implementation'),

        html.Br(),
        html.Label('Solver'),
        dcc.Dropdown(['svrg', 'qning-svrg', 'qning-miso', 'qning-ista', 'miso', 'ista', 'fista', 'catalyst-svrg', 'catalyst-miso', 'catalyst-ista', 'acc-svrg', 'auto', 'all'], 'qning-miso', id='solver', style={'marginBottom': 50})],
                style={'marginLeft': 125, 'marginTop': 50, 'width': "15%"}),
        ]),
        ],
    )

# ---------------------- Callbacks ---------------------------------
# Callbacks are all client-side (https://dash.plot.ly/performance)
# in order to transform the app into static html pages
# javascript functions are defined in assets/callbacks.js

app.clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='update_store_data'
    ),
    output=Output('plot', 'figure'),
    inputs=[
        Input('store', "data"),
        Input('dataset', 'value'),
        Input('core', 'value'),
        Input('implementation', 'value'),
        Input('solver', 'value'),
        Input('log-lin', 'value')
        ],
    state=[State('store', 'data')],
    )


if __name__ == '__main__':
    app.run_server(debug=debug)