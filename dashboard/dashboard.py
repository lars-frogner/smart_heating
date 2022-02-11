import os
import sys
import glob
import pathlib
import pickle
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as ps
import plotly.express as px
import matplotlib
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template


def mpl_to_plotly_color(mpl_color):
    if mpl_color is None:
        return None
    else:
        return matplotlib.colors.to_hex(mpl_color)


def mpl_to_plotly_linestyle(mpl_ls):
    return {'-': 'solid', '--': 'dash', ':': 'dot'}[mpl_ls]


def mpl_to_plotly_scatter_size(mpl_s):
    return 10 * np.sqrt(mpl_s / 10)


def read_data(input_path):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    return data


def find_figure_data():
    current_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    data_paths = {}
    for path in glob.iglob('*/figure_data/*.pickle',
                           root_dir=current_dir.parent):
        path = pathlib.Path(path)
        app_name = path.parts[0]
        figure_name = path.stem
        if app_name not in data_paths:
            data_paths[app_name] = {}
        data_paths[app_name][figure_name] = current_dir / path
    return data_paths


def mpl_to_plotly_vline(mpl_artist):
    return dict(is_vline=True,
                x=mpl_artist['x_value'],
                exclude_empty_subplots=False,
                layer='below',
                line=dict(
                    color=mpl_to_plotly_color(mpl_artist.get('color', None)),
                    dash=mpl_to_plotly_linestyle(mpl_artist.get('ls', '-'))))


def mpl_to_plotly_hline(mpl_artist):
    return dict(is_hline=True,
                y=mpl_artist['y_value'],
                exclude_empty_subplots=False,
                layer='below',
                line=dict(
                    color=mpl_to_plotly_color(mpl_artist.get('color', None)),
                    dash=mpl_to_plotly_linestyle(mpl_artist.get('ls', '-'))))


def mpl_to_plotly_vrect(mpl_artist):
    x = mpl_artist['x_values']
    y = mpl_artist['y_values']
    y_threshold = mpl_artist['y_threshold']

    filled = y > y_threshold

    all_x0_indices = list(np.nonzero(filled[1:] > filled[:-1])[0] + 1)
    if filled[0]:
        all_x0_indices.insert(0, 0)

    all_x1_indices = list(np.nonzero(filled[1:] < filled[:-1])[0])
    if filled[-1]:
        all_x1_indices.append(-1)

    return dict(is_vrect=True,
                vrects=[
                    dict(x0=x[x0_idx],
                         x1=x[x1_idx],
                         exclude_empty_subplots=False,
                         layer='below',
                         fillcolor=mpl_to_plotly_color(
                             mpl_artist.get('color', None)),
                         opacity=mpl_artist.get('alpha', 1.0),
                         line=dict(width=0))
                    for x0_idx, x1_idx in zip(all_x0_indices, all_x1_indices)
                ])


def mpl_to_plotly_line(mpl_artist):
    marker = mpl_artist.get('marker', None)

    name = mpl_artist.get('label', None)

    return dict(type='scatter',
                x=mpl_artist['x_values'],
                y=mpl_artist['y_values'],
                mode='lines' + ('+markers' if marker is not None else ''),
                line=dict(
                    color=mpl_to_plotly_color(mpl_artist.get('color', None)),
                    dash=mpl_to_plotly_linestyle(mpl_artist.get('ls', '-'))),
                marker=(None if marker is None else dict(size={
                    'o': 11,
                    '.': 7
                }[marker])),
                name=name,
                showlegend=(name is not None),
                secondary_y=(mpl_artist.get('ax', 0) == 1))


def mpl_to_plotly_scatter(mpl_artist):
    color = mpl_artist.get('c', mpl_artist.get('color', None))
    multiple_colors = isinstance(color, (np.ndarray, list))
    if not multiple_colors:
        color = mpl_to_plotly_color(color)

    name = mpl_artist.get('label', None)

    return dict(type='scatter',
                x=mpl_artist['x_values'],
                y=mpl_artist['y_values'],
                mode='markers',
                marker=dict(
                    color=color,
                    size=mpl_to_plotly_scatter_size(mpl_artist.get('s', 10)),
                    colorbar=(dict(title=mpl_artist.get('clabel', ''))
                              if mpl_artist.get('colorbar', False) else None),
                    colorscale=mpl_artist.get('cmap_name', 'viridis')),
                name=name,
                showlegend=(name is not None),
                secondary_y=(mpl_artist.get('ax', 0) == 1))


def mpl_artist_to_plotly_graph_object(mpl_artist):
    artist_type = mpl_artist.get('type', 'plot')
    if artist_type == 'plot':
        return mpl_to_plotly_line(mpl_artist)
    elif artist_type == 'scatter':
        return mpl_to_plotly_scatter(mpl_artist)
    elif artist_type == 'vline':
        return mpl_to_plotly_vline(mpl_artist)
    elif artist_type == 'hline':
        return mpl_to_plotly_hline(mpl_artist)
    elif artist_type == 'fill':
        return mpl_to_plotly_vrect(mpl_artist)


def mpl_to_plotly_layout(mpl_layout):
    aspect_ratio = mpl_layout.get('aspect', None)
    if aspect_ratio == 'equal':
        aspect_ratio = 1.0
    return dict(title_text=mpl_layout.get('title', None),
                xaxis_title=mpl_layout.get('xlabel', None),
                yaxis_title=mpl_layout.get('ylabel', None),
                secondary_yaxis_title=mpl_layout.get('second_ylabel', None),
                showlegend=(mpl_layout.get('legend_loc', None) is not None),
                aspect_ratio=aspect_ratio,
                legend_orientation='v',
                template='darkly',
                margin=dict(l=70, r=70, t=60, b=50))


def create_figure(*graph_objects, layout={}):
    uses_secondary_y = len(
        list(filter(lambda g: g.get('secondary_y', False), graph_objects))) > 0
    fig = ps.make_subplots(specs=[[{"secondary_y": uses_secondary_y}]])

    used_names = []
    for graph_object in graph_objects:
        if graph_object.pop('is_vrect', False):
            for vrect in graph_object['vrects']:
                fig.add_vrect(**vrect)
        elif graph_object.pop('is_vline', False):
            fig.add_vline(**graph_object)
        elif graph_object.pop('is_hline', False):
            fig.add_hline(**graph_object)
        else:
            secondary_y = graph_object.pop('secondary_y', False)

            name = graph_object.get('name', None)
            if name in used_names:
                graph_object['showlegend'] = False
            elif name is not None:
                used_names.append(name)

            fig.add_trace(graph_object, secondary_y=secondary_y)

    aspect_ratio = layout.pop('aspect_ratio', None)
    if aspect_ratio is not None:
        fig.update_yaxes(scaleanchor='x', scaleratio=aspect_ratio)
    if uses_secondary_y:
        fig.update_yaxes(scaleanchor='x',
                         scaleratio=aspect_ratio,
                         secondary_y=True)

    secondary_yaxis_title = layout.pop('secondary_yaxis_title', None)
    if secondary_yaxis_title is not None:
        fig.update_yaxes(title_text=secondary_yaxis_title, secondary_y=True)

    if uses_secondary_y:
        fig.update_yaxes(showgrid=False, secondary_y=True)

    fig.update_layout(layout)

    return fig


def mpl_to_plotly_figure(data):
    artists = data.pop('artists')
    # for artist in artists:
    #     print(artist.get('type', None))
    sorted_artists = []
    for type_name in ['fill', 'hline', 'vline', 'scatter', 'plot']:
        for artist in artists:
            if artist.get('type', 'plot') == type_name and artist.get('ax',
                                                                      0) == 0:
                # if artist.get('label',
                #               None) not in ('Temp. forecast', 'Temp. outside',
                #                             'Temp. inside'):
                sorted_artists.append(artist)
        for artist in artists:
            if artist.get('type', 'plot') == type_name and artist.get('ax',
                                                                      0) == 1:
                sorted_artists.append(artist)

    # for artist in sorted_artists:
    #     print(artist.get('type', None))
    # print(len(artists), len(sorted_artists))

    graph_objects = filter(lambda obj: obj is not None,
                           (mpl_artist_to_plotly_graph_object(artist)
                            for artist in sorted_artists))
    return create_figure(*graph_objects, layout=mpl_to_plotly_layout(data))


def create_figure_from_data_path(data_path):
    return mpl_to_plotly_figure(read_data(data_path))


def create_app():
    load_figure_template('darkly')

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

    data_paths = find_figure_data()
    app_names = sorted(list(data_paths.keys()))

    app.layout = dbc.Container([
        dbc.Row(dbc.Col(html.H1('Smart heating', className='my-3'))),
        dbc.Row([
            dbc.Col(html.Label('Room'),
                    className='col-auto d-flex align-items-center pe-0'),
            dbc.Col(dcc.Dropdown(app_names,
                                 app_names[0],
                                 clearable=False,
                                 id='app-name-dropdown'),
                    className='col-5')
        ]),
        dbc.Container(id='figure-container', className='px-0')
    ])

    @app.callback(dash.Output('figure-container', 'children'),
                  dash.Input('app-name-dropdown', 'value'))
    def update_selected_app(app_name):
        app_data_paths = data_paths[app_name]
        return [
            dbc.Row(dbc.Col(
                dcc.Graph(id=f'{name}-fig',
                          figure=create_figure_from_data_path(
                              app_data_paths[name]))),
                    className='mt-3') for name in sorted(app_data_paths)
        ]

    return app


if __name__ == '__main__':
    app = create_app()
    app.run_server(debug=True)
