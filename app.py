import dash
import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
import plotly.express as px
import statsmodels
import xlrd




# archivo de datos
datafile = 'datosRL.xls'
# sacando datos del archivo hacia un dataframe
data = pd.read_excel(datafile, header=1)
print(data)
data2 = data.transpose()
data2.dropna(axis=0, inplace=True)
data2.rename(columns=data2.iloc[0], inplace=True)
data2.drop(['Año'], inplace=True)
print(data2)
print(data2.describe())


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([html.H1('Regresion Lineal TP Final Luis Marquez', style={'color': 'grey',
                                                                                'text-align': 'center'
                                                                                }),
                       html.Div(dcc.Dropdown(id='demo-dropdown',
                                             options=[{'label': 'Grafica de Dispersión sola', 'value': 'fig'},
                                                      {'label': 'Grafica con Linea de Regresión', 'value': 'fig2'}],
                                             value='fig'), style={'color': 'lightblue',
                                                                        'text-align': 'center',
                                                                        'display': 'inline-block',
                                                                        'width': '40%'
                                                                        }),
                       dcc.Graph(id='Grafica Seleccionada',
                                 style={'text-align': 'center','width':'80%'})
                       ])

@app.callback(
    dash.dependencies.Output('Grafica Seleccionada', 'figure'),
    [dash.dependencies.Input('demo-dropdown', 'value')])
def update_figure(selected_value):

    fig = px.scatter(data2, x='Renta Neta', y='consumo')
    fig2 = px.scatter(data2, x='Renta Neta', y='consumo', trendline="ols")
    dic = {'fig': fig, 'fig2': fig2}
    figure=dic.get(selected_value)
    return figure


if __name__ == '__main__':
    app.run_server(debug=True)
