import os
import sys
import glob
import re
import pathlib
import pickle
import collections
import numpy as np
import ruamel.yaml
import plotly.graph_objects as go
import plotly.subplots as ps
import plotly.express as px
import matplotlib
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

yaml = ruamel.yaml.YAML()


def read_pickle(input_path):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    return data


def read_lines(input_path, n_last_lines=None, position_pointer=None):
    lines = collections.deque(maxlen=n_last_lines)

    with open(input_path, 'r') as f:
        if position_pointer is not None:
            f.seek(position_pointer)

        for line_num, line in enumerate(f):
            lines.append(line)

        position_pointer = f.tell()

    return list(lines), position_pointer


def read_yaml(input_path):
    with open(input_path, 'r') as f:
        data = yaml.load(f)
    return data


def write_yaml(output_path, data):
    with open(output_path, 'w') as f:
        yaml.dump(data, f)


def mpl_to_plotly_color(mpl_color):
    if mpl_color is None:
        return None
    else:
        return matplotlib.colors.to_hex(mpl_color)


def mpl_to_plotly_linestyle(mpl_ls):
    return {'-': 'solid', '--': 'dash', ':': 'dot'}[mpl_ls]


def mpl_to_plotly_scatter_size(mpl_s):
    return 10 * np.sqrt(mpl_s / 10)


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
    return mpl_to_plotly_figure(read_pickle(data_path))


class SmartHeatingController:
    SCRIPT_DIR = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    LOG_DIR = SCRIPT_DIR.parent.parent.parent / 'log'
    MAIN_LOG_PATH = LOG_DIR / 'appdaemon.log'
    ERROR_LOG_PATH = LOG_DIR / 'error.log'
    CONFIG_DIR = SCRIPT_DIR.parent / 'config'

    ENTITY_ID_NAMES = [
        'thermostat_id', 'heater_id', 'weather_id', 'outside_thermometer_id',
        'power_price_id', 'power_consumption_meter_id'
    ]
    ENTITY_ID_LABELS = [
        'Thermostat', 'Heater', 'Weather service', 'Outdoor thermometer',
        'Electricity price', 'Heater power consump.'
    ]

    def __init__(self):
        self.room_names = self.obtain_room_names()
        self.valid_room_name_regex = re.compile('^[a-z_]+$')

    @property
    def main_log_path(self):
        return self.__class__.MAIN_LOG_PATH

    @property
    def error_log_path(self):
        return self.__class__.ERROR_LOG_PATH

    def obtain_room_names(self):
        return self.__class__.find_room_names()
    
    def add_room(self, room_name):
        self.room_names.append(room_name)
        self.room_names.sort()
    
    def room_name_is_valid(self, room_name):
        return self.valid_room_name_regex.match(room_name) is not None

    def obtain_config(self, room_name):
        config_path = self.__class__.CONFIG_DIR / f'{room_name}.yaml'
        if config_path.exists():
            return read_yaml(config_path)[room_name]
        else:
            return {
                'module': 'smart_heating',
                'class': 'SmartHeating',
                'tibber_access_token': '!secret tibber_access_token'
            }

    def save_config(self, room_name, config):
        config_path = self.__class__.CONFIG_DIR / f'{room_name}.yaml'
        write_yaml(config_path, {room_name: config})

    def set_log_position_pointer(self, log_name, position_pointer):
        setattr(self, f'{log_name}_log_position_pointer', position_pointer)

    def get_log_position_pointer(self, log_name):
        return getattr(self, f'{log_name}_log_position_pointer')

    def set_current_app(self, room_name):
        self.current_room_name = room_name

    def get_current_app(self):
        return getattr(self, 'current_room_name', None)

    def set_figure_data_modification_time(self, room_name, figure_name,
                                          modification_time):
        setattr(self, f'{room_name}_{figure_name}_data_modification_time',
                modification_time)

    def get_figure_data_modification_time(self,
                                          room_name,
                                          figure_name,
                                          default=None):
        return getattr(self,
                       f'{room_name}_{figure_name}_data_modification_time',
                       default)

    @classmethod
    def get_figure_data_path(cls, room_name, figure_name=None):
        dir_path = cls.SCRIPT_DIR.parent / room_name / 'figure_data'
        if figure_name is None:
            return dir_path
        else:
            return dir_path / f'{figure_name}.pickle'

    @classmethod
    def find_room_names(cls):
        return sorted([path.stem for path in cls.CONFIG_DIR.glob('*.yaml')])

    @classmethod
    def find_figure_names(cls, room_name):
        return sorted([
            path.stem
            for path in cls.get_figure_data_path(room_name).glob('*.pickle')
        ])

    @classmethod
    def find_figure_names_and_data_paths(cls, room_name):
        figure_names = cls.find_figure_names(room_name)
        return figure_names, [
            cls.get_figure_data_path(room_name, figure_name)
            for figure_name in figure_names
        ]


def create_dashboard():
    load_figure_template('darkly')

    controller = SmartHeatingController()

    def create_log_switch():
        return dbc.Row(
            dbc.Col(dbc.Switch(id="log-switch", label="Show", value=False)))

    def create_logs():
        return dbc.Container([
            dbc.Row(dbc.Col(html.Label('Main log')), className='mb-2'),
            dbc.Container(html.Div(id='main-log-container'),
                          className='log-container p-1 mb-3'),
            dbc.Row(dbc.Col(html.Label('Error log')), className='my-2'),
            dbc.Container(html.Div(id='error-log-container'),
                          className='log-container p-1 mb-3'),
        ],
                             id='log-container',
                             className='px-0 mb-4',
                             style=dict(display='none'))

    def create_room_select(selected_room_name=None):
        room_names = controller.room_names
        if selected_room_name is not None:
            assert selected_room_name in room_names
        if len(room_names) == 0:
            return None
        else:
            return dbc.Select(id='room-select',
                                      value=(room_names[0] if selected_room_name is None else selected_room_name),
                                      options=[
                                          dict(label=room_name,
                                               value=room_name)
                                          for room_name in room_names
                                      ])

    def create_room_menu():
        return dbc.Container([
            dbc.Row(dbc.Col(html.H3('Room'), className='mt-3 mb-2')),
            dbc.Row([
                dbc.Col(create_room_select(), id='room-select-container',
                           className='col-auto me-3'),
                dbc.Col(
                    dbc.Row([
                        dbc.Col(
                            [dbc.Input(id='new-room-input',
                                      placeholder='New room name'),
                            dbc.FormFeedback(
                        'Room name must be unique with only letters and underscores',
                        type='invalid',
                    )], className='col-auto pe-0'),
                        dbc.Col(
                            dbc.Button('Add new',
                                       id='new-room-button',
                                       disabled=True), className='col-auto')
                    ]))
            ])
        ],
                             className='px-0 mb-4')

    dashboard = dash.Dash(__name__,
                          external_stylesheets=[dbc.themes.DARKLY],
                          suppress_callback_exceptions=True)

    dashboard.layout = dbc.Container([
        dbc.Row(dbc.Col(html.H1('Smart heating', className='mt-3 mb-4'))),
        dcc.Interval(id='log-interval', interval=10000, disabled=True),
        dbc.Row(dbc.Col(html.H3('Logs')), className='mb-2'),
        create_log_switch(),
        create_logs(),
        create_room_menu(),
        dbc.Row(dbc.Col(html.H4('Configuration')), className='mt-3'),
        dcc.Loading(dbc.Container(id='config-container',
                                  className='px-0 mt-3')),
        dbc.Row(dbc.Col(html.H4('Figures')), className='mt-3'),
        dcc.Interval(id='figure-interval', interval=600000),
        dcc.Loading(dbc.Container(id='figure-container',
                                  className='px-0 mb-4'))
    ])

    @dashboard.callback(dash.Output('new-room-button', 'disabled'),
                        dash.Output('new-room-input', 'invalid'),
                        dash.Input('new-room-input', 'value'),
                        prevent_initial_call=True)
    def enable_or_disable_new_room_button(new_room_name):
        invalid = not controller.room_name_is_valid(new_room_name)
        return invalid, invalid and new_room_name != ''

    @dashboard.callback(dash.Output('room-select-container', 'children'),
                        dash.Input('new-room-button', 'n_clicks'),
                        dash.State('new-room-input', 'value'),
                        prevent_initial_call=True)
    def add_room(n_clicks, new_room_name):
        controller.add_room(new_room_name)
        return create_room_select(selected_room_name=new_room_name)

    @dashboard.callback(dash.Output('log-interval', 'disabled'),
                        dash.Input('log-switch', 'value'),
                        prevent_initial_call=True)
    def toggle_interval(switch_on):
        return not switch_on

    @dashboard.callback(dash.Output('log-container', 'style'),
                        dash.Input('log-switch', 'value'),
                        prevent_initial_call=True)
    def toggle_log_visibility(switch_on):
        return dict(display='block') if switch_on else dict(display='none')

    def update_log(log_name, log_path, existing_lines, n_last_lines=100):
        if existing_lines is None:
            lines, log_position_pointer = read_lines(log_path,
                                                     n_last_lines=n_last_lines)
            controller.set_log_position_pointer(log_name, log_position_pointer)
            return list(map(html.Div, lines))
        else:
            new_lines, log_position_pointer = read_lines(
                log_path,
                position_pointer=controller.get_log_position_pointer(log_name))
            controller.set_log_position_pointer(log_name, log_position_pointer)
            return existing_lines + list(map(html.Div, new_lines))

    @dashboard.callback(dash.Output('main-log-container', 'children'),
                        dash.Input('log-interval', 'n_intervals'),
                        dash.State('main-log-container', 'children'))
    def update_main_log(n_intervals, existing_lines):
        return update_log('main', controller.main_log_path, existing_lines)

    @dashboard.callback(dash.Output('error-log-container', 'children'),
                        dash.Input('log-interval', 'n_intervals'),
                        dash.State('error-log-container', 'children'))
    def update_error_log(n_intervals, existing_lines):
        return update_log('error', controller.error_log_path, existing_lines)

    @dashboard.callback(dash.Output('figure-container', 'children'),
                        dash.Input('figure-interval', 'n_intervals'),
                        dash.Input('room-select', 'value'),
                        dash.State('figure-container', 'children'))
    def update_selected_room_figures(n_intervals, room_name, old_figures):
        if room_name is None:
            raise dash.exceptions.PreventUpdate

        new_app = room_name != controller.get_current_app(
        ) or old_figures is None
        controller.set_current_app(room_name)

        figures = []
        for idx, (figure_name, data_path) in enumerate(
                zip(*SmartHeatingController.find_figure_names_and_data_paths(
                    room_name))):

            modification_time = os.path.getmtime(data_path)

            if new_app or modification_time > controller.get_figure_data_modification_time(
                    room_name, figure_name, default=modification_time):
                figures.append(
                    dbc.Row(dbc.Col(
                        dcc.Graph(
                            id=f'{figure_name}-fig',
                            figure=create_figure_from_data_path(data_path))),
                            className='mt-3',
                            id=figure_name))

                controller.set_figure_data_modification_time(
                    room_name, figure_name, modification_time)
            else:
                for idx, fig in enumerate(old_figures):
                    if fig['props']['id'] == figure_name:
                        figures.append(old_figures.pop(idx))
                        break

        return figures

    def create_temperature_slider(room_config):
        comfort_temperature = room_config.get('comfort_temperature', 18.0)
        minimum_temperature = room_config.get('minimum_temperature', 6.0)
        maximum_temperature = room_config.get('maximum_temperature', 25.0)
        mark_temperatures = [0, 5, 10, 15, 20, 25, 30]
        return dbc.Container([
            dbc.Row(dbc.Col(html.H5('Temperatures')), className='mt-3 mb-2'),
            dbc.Row([
                dbc.Col(dbc.Label('Min', id='min-temp-text'),
                        className='col-4'),
                dbc.Col(dbc.Label('Comfort', id='comfort-temp-text'),
                        className='col-4 text-center'),
                dbc.Col(dbc.Label('Max', id='max-temp-text'),
                        className='col-4 text-end')
            ],
                    className='mb-2'),
            dbc.Row(dbc.Col(
                dcc.RangeSlider(id='temp-slider',
                                min=0,
                                max=30,
                                step=0.5,
                                marks={t: f'{t}°C'
                                       for t in mark_temperatures},
                                value=[
                                    minimum_temperature, comfort_temperature,
                                    maximum_temperature
                                ],
                                pushable=0.5,
                                tooltip=dict(always_visible=True,
                                             placement='top'))),
                    className='pt-2')
        ],
                             className='px-0 ms-0 me-auto',
                             id='temp-container')

    def create_comfort_period_inputs(room_config):
        comfort_start_time = room_config.get('comfort_start_time', '19:00')
        comfort_end_time = room_config.get('comfort_end_time', '07:00')
        pattern = '([01][0-9]|2[0-3]):([0-5][0-9])'
        return dbc.Container([
            dbc.Row(dbc.Col(html.H5('Comfort period')), className='mt-3 mb-2'),
            dbc.Row([
                dbc.Col(dbc.Label('Start', className='mb-0'),
                        className='col-auto d-flex align-items-center pe-0'),
                dbc.Col([
                    dbc.Input(id='comfort_start_time-input',
                              type='text',
                              value=comfort_start_time,
                              placeholder='HH:MM',
                              pattern=pattern,
                              className='time-input'),
                    dbc.FormFeedback(
                        'Required',
                        type='invalid',
                    )
                ],
                        className='col-auto'),
                dbc.Col(dbc.Label('End', className='mb-0'),
                        className='col-auto d-flex align-items-center pe-0'),
                dbc.Col([
                    dbc.Input(id='comfort_end_time-input',
                              type='text',
                              value=comfort_end_time,
                              placeholder='HH:MM',
                              pattern=pattern,
                              className='time-input'),
                    dbc.FormFeedback(
                        'Required',
                        type='invalid',
                    )
                ],
                        className='col-auto')
            ])
        ],
                             className='px-0')

    def create_mode_select(room_config):
        mode = room_config.get('mode', 'optimal')
        return dbc.Container([
            dbc.Row(dbc.Col(html.H5('Mode')), className='mt-3 mb-2'),
            dbc.Row(
                dbc.Col(dbc.Select(id='mode-select',
                                   value=mode,
                                   options=[
                                       dict(label='Classic', value='classic'),
                                       dict(label='Smart', value='smart'),
                                       dict(label='Optimal', value='optimal')
                                   ]),
                        className='col-auto'))
        ],
                             className='px-0')

    def create_heating_power_input(room_config):
        heating_power = room_config.get('heating_power', 1000)
        return dbc.Container([
            dbc.Row(dbc.Col(html.H5('Heating power')), className='mt-3 mb-2'),
            dbc.Row([
                dbc.Col([
                    dbc.Input(id='heating_power-input',
                              type='number',
                              min=0,
                              max=3000,
                              step=10,
                              value=heating_power,
                              className='number-input'),
                    dbc.FormFeedback(
                        'Required',
                        type='invalid',
                    )
                ],
                        className='col-auto pe-0 me-2'),
                dbc.Col(
                    dbc.Label('watts', className='mb-0'),
                    className='col-auto d-flex align-items-center px-0 ms-0')
            ])
        ],
                             className='px-0')

    def create_plot_interval_input(room_config):
        plot_interval = room_config.get('plot_interval', 600)
        return dbc.Container([
            dbc.Row(dbc.Col(html.H5('Plotting interval')),
                    className='mt-3 mb-2'),
            dbc.Row([
                dbc.Col([
                    dbc.Input(id='plot_interval-input',
                              type='number',
                              min=0,
                              step=10,
                              value=plot_interval,
                              className='number-input'),
                    dbc.FormFeedback(
                        'Required',
                        type='invalid',
                    )
                ],
                        className='col-auto pe-0 me-2'),
                dbc.Col(
                    dbc.Label('seconds', className='mb-0'),
                    className='col-auto d-flex align-items-center px-0 ms-0')
            ])
        ],
                             className='px-0')

    def create_options_checklist(room_config):
        run = room_config.get('run', True)
        debug = room_config.get('debug', False)
        hard_reset = room_config.get('hard_reset', False)
        value_map = dict(run=run, hard_reset=hard_reset, debug=debug)
        values = [name for name, value in value_map.items() if value]
        return dbc.Container([
            dbc.Row(dbc.Col(html.H5('Options')), className='mt-3 mb-2'),
            dbc.Row(
                dbc.Col(dbc.Checklist(
                    id='options-checklist',
                    value=values,
                    options=[
                        dict(label='Control heating', value='run'),
                        dict(label='Rebuild model', value='hard_reset'),
                        dict(label='Log debug messages', value='debug')
                    ],
                    switch=True),
                        className='col-auto'))
        ],
                             className='px-0')

    def create_entity_id_inputs(room_config):
        rows = []
        for name, label in zip(controller.ENTITY_ID_NAMES,
                               controller.ENTITY_ID_LABELS):
            value = room_config.get(name, '')
            rows.append(
                dbc.Row([
                    dbc.Col(
                        dbc.Label(label, className='entity-id-label mb-0'),
                        className='col-auto d-flex align-items-center pe-0'),
                    dbc.Col([
                        dbc.Input(id=f'{name}-input',
                                  type='text',
                                  value=value,
                                  className='entity-id-input'),
                        dbc.FormFeedback(
                            'Required',
                            type='invalid',
                        )
                    ],
                            className='col-auto pe-0 me-2')
                ]))

        return dbc.Container([
            dbc.Row(dbc.Col(html.H5('Entity IDs')), className='mt-3 mb-2'),
            *rows
        ],
                             className='px-0')

    def create_config_submit_button():
        return dbc.Container(dbc.Row(
            dbc.Col(dbc.Button('Submit configuration',
                               id='config-submit-button',
                               color='primary'),
                    className='col-auto')),
                             className='px-0 my-4')

    @dashboard.callback(dash.Output('config-container', 'children'),
                        dash.Output('figure-interval', 'interval'),
                        dash.Input('room-select', 'value'))
    def update_selected_room_config(room_name):
        if room_name is None:
            raise dash.exceptions.PreventUpdate

        config = controller.obtain_config(room_name)
        return dbc.Form([
            create_temperature_slider(config),
            dbc.Row([
                dbc.Col(create_comfort_period_inputs(config),
                        className='col-auto me-4 mt-2'),
                dbc.Col(create_mode_select(config),
                        className='col-auto me-4 mt-2'),
                dbc.Col(create_heating_power_input(config),
                        className='col-auto me-4 mt-2'),
                dbc.Col(create_plot_interval_input(config),
                        className='col-auto me-4 mt-2'),
                dbc.Col(create_options_checklist(config),
                        className='col-auto me-4 mt-2')
            ]),
            create_entity_id_inputs(config),
            create_config_submit_button()
        ]), config.get('plot_interval', 600) * 1000

    entity_id_input_ids = [
        f'{name}-input' for name in controller.ENTITY_ID_NAMES
    ]
    other_input_ids = [
        'comfort_start_time-input', 'comfort_end_time-input',
        'heating_power-input', 'plot_interval-input'
    ]
    all_input_ids = other_input_ids + entity_id_input_ids

    invalid_outputs = [dash.Output(id, 'invalid') for id in all_input_ids]
    input_states = [dash.State(id, 'value') for id in all_input_ids]

    @dashboard.callback(*invalid_outputs,
                        dash.Input('config-submit-button', 'n_clicks'),
                        dash.State('room-select', 'value'),
                        dash.State('temp-slider', 'value'),
                        dash.State('mode-select', 'value'),
                        dash.State('options-checklist', 'value'),
                        *input_states,
                        prevent_initial_call=True)
    def build_configuration_from_form(n_clicks, room_name, temperatures, mode,
                                      options, *input_values):
        invalid = [value in ('', None) for value in input_values]
        if mode == 'classic':
            invalid[-5:] = [False] * 5
        else:
            invalid[-3:] = [False] * 3

        if not np.any(invalid):
            config = controller.obtain_config(room_name)

            config['minimum_temperature'] = temperatures[0]
            config['comfort_temperature'] = temperatures[1]
            config['maximum_temperature'] = temperatures[2]

            config['mode'] = mode

            config['run'] = 'run' in options
            config['hard_reset'] = 'hard_reset' in options
            config['debug'] = 'debug' in options

            for input_id, input_value in zip(all_input_ids, input_values):
                name = input_id[:-6]
                if input_value in ('', None):
                    config.pop(name, None)
                else:
                    config[name] = input_value

            controller.save_config(room_name, config)

        return invalid

    return dashboard


if __name__ == '__main__':
    dashboard = create_dashboard()
    dashboard.run_server(debug=True)
