import os
import sys
import asyncio
import pickle
import shutil
import datetime as dt
import pytz
from typing import Union, Any, Optional, Tuple, Callable, Iterable
from pathlib import Path
import numpy as np
import scipy.interpolate
import scipy.integrate
import scipy.signal
import sklearn.linear_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import tibber
import hassapi as hass


def plot(*artists,
         x_uses_datetimes=False,
         xlim=None,
         ylim=None,
         xlabel=None,
         ylabel=None,
         title=None,
         second_ylim=None,
         second_ylabel=None,
         legend_loc='best',
         aspect=None,
         fig_kwargs={},
         data_save_path=None,
         output_path=None):

    if data_save_path is not None:
        data = dict(artists=artists,
                    x_uses_datetimes=x_uses_datetimes,
                    xlim=xlim,
                    ylim=ylim,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    title=title,
                    second_ylim=second_ylim,
                    second_ylabel=second_ylabel,
                    legend_loc=legend_loc,
                    aspect=aspect,
                    fig_kwargs=fig_kwargs)

        with open(data_save_path, 'wb') as f:
            pickle.dump(data, f)

    if output_path is None:
        return

    fig, first_ax = plt.subplots(**fig_kwargs)

    if x_uses_datetimes:
        fig.autofmt_xdate()
        first_ax.xaxis.set_major_formatter(
            mdates.DateFormatter('%d.%m-%y %H:%M'))

    if aspect is not None:
        first_ax.set_aspect(aspect)

    first_ax.set_xlabel(xlabel)
    first_ax.set_ylabel(ylabel)
    first_ax.set_title(title)

    if xlim is not None:
        first_ax.set_xlim(*xlim)
    if ylim is not None:
        first_ax.set_ylim(*ylim)

    axes = [first_ax]

    def get_ax(idx):
        if idx == 1 and len(axes) == 1:
            axes.append(axes[0].twinx())
        return axes[idx]

    if second_ylabel is not None:
        get_ax(1).set_ylabel(second_ylabel)

    if second_ylim is not None:
        get_ax(1).set_ylim(*second_ylim)

    legend_handles = []
    legend_labels = []

    auto_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    auto_color_idx = 0

    for artist in artists:
        ax = get_ax(artist.pop('ax', 0))

        artist_type = artist.pop('type', 'plot')

        if artist_type == 'fill':
            x_values = artist.pop('x_values')
            y_values = artist.pop('y_values')
            y_threshold = artist.pop('y_threshold')
            ax.fill_between(x_values,
                            0,
                            1,
                            where=(y_values > y_threshold),
                            transform=ax.get_xaxis_transform(),
                            **artist)
        elif artist_type == 'vline':
            x_value = artist.pop('x_value')
            ax.axvline(x_value, **artist)
        elif artist_type == 'hline':
            y_value = artist.pop('y_value')
            ax.axhline(y_value, **artist)
        elif artist_type in ('plot', 'scatter'):
            x_values = artist.pop('x_values')
            y_values = np.asfarray(artist.pop('y_values'))
            c = artist.get('c', None)
            color = artist.get('color', None)
            label = artist.pop('label', None)

            if color is None and c is None:
                artist['color'] = auto_colors[auto_color_idx %
                                              len(auto_colors)]
                auto_color_idx += 1

            if artist_type == 'plot':
                ax.plot(x_values, y_values, **artist)

                if label is not None and label not in legend_labels:
                    legend_handles.append(Line2D([], [], **artist))
                    legend_labels.append(label)
            else:
                colorbar = artist.pop('colorbar', False)
                clabel = artist.pop('clabel', None)

                sc = ax.scatter(x_values, y_values, **artist)

                if colorbar:
                    fig.colorbar(sc, label=clabel)

                if label is not None and label not in legend_labels:
                    legend_handles.append(
                        Line2D([], [],
                               ls='',
                               marker='o',
                               color=sc.get_facecolor(),
                               alpha=sc.get_alpha()))
                    legend_labels.append(label)

    if len(legend_handles) > 0:
        first_ax.legend(legend_handles, legend_labels, loc=legend_loc)

    fig.tight_layout()
    fig.savefig(output_path)


def datetime_to_timestamp(datetime):
    return datetime.timestamp()


def datetimes_to_timestamps(datetimes):
    return np.fromiter(map(datetime_to_timestamp, datetimes), float)


def timestamp_to_datetime(timestamp):
    return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)


def timestamps_to_datetimes(timestamps):
    return list(map(timestamp_to_datetime, timestamps))


def next_datetime_for_time(current_datetime, time):
    if time >= current_datetime.time():
        next_datetime = dt.datetime.combine(current_datetime.date(), time)
    else:
        next_datetime = dt.datetime.combine(
            current_datetime.date() + dt.timedelta(days=1), time)
    return next_datetime.replace(tzinfo=current_datetime.tzinfo)


def remaining_seconds_to_time(current_datetime, time):
    return (next_datetime_for_time(current_datetime, time) -
            current_datetime).total_seconds()


def delete_all_files_in_folder(folder_path):
    for file_path in folder_path.glob('**/*'):
        if file_path.is_file():
            file_path.unlink()
        elif file_path.is_dir():
            shutil.rmtree(file_path)


class SmartHeating(hass.Hass):

    def initialize(self):
        self.valid_modes = ['classic', 'smart', 'optimal']
        self.debug = None
        self.plot_interval = None
        self.mode = None
        self.setup_time = None
        self.comfort_start_time = None
        self.comfort_end_time = None
        self.comfort_temperature = None
        self.minimum_temperature = None
        self.thermostat = None
        self.heater = None
        self.outside_thermometer = None
        self.weather = None
        self.power_consumption_meter = None
        self.heating_power = None
        self.tibber_access_token = None

        self.temperature_forecast = None
        self.temperature_forecast_end_time = None

        self.power_price = None

        self.modeling_time_interval = dt.timedelta(minutes=0.1).total_seconds()
        self.modeling_buffer_time = 0.5 * dt.timedelta(hours=1).total_seconds()

        self.scheduled_heating_period = None
        self.next_monitoring_time = None

        self._tz = pytz.timezone(self.get_timezone())

        self._init_paths()
        self._handle_input_args()

        if self.debug:
            self.set_log_level('DEBUG')

        if self.rebuild_model:
            self.delete_all_data()
            self.run_in(self.gather_data_for_initial_model, 0)
        elif self.control_heating:
            self.run_in(self.start_running, 0)

        if self.keep_model_updated:
            self.schedule_model_updates()

        if self.plot_interval is not None:
            self.schedule_plot_data_updates()

        self.log('SmartHeating initialized', level='INFO')

    def _init_paths(self):
        self.root_path = Path(self.app_dir) / 'smart_heating' / self.name
        self.data_path = self.root_path / 'data'
        self.figure_path = self.root_path / 'figures'
        self.figure_data_path = self.root_path / 'figure_data'

        self.setup_time_path = self.data_path / 'setup_time.dat'

        self.table_path = self.data_path / 'table.npz'
        self.model_path = self.data_path / 'model.npz'
        self.forecast_correction_path = self.data_path / 'forecast_correction.npz'

        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.figure_path, exist_ok=True)
        os.makedirs(self.figure_data_path, exist_ok=True)

    def _handle_input_args(self):
        self.debug = self.args.get('debug', False)
        self.control_heating = self.args.get('control_heating', True)
        self.rebuild_model = self.args.get('rebuild_model', False)
        self.keep_model_updated = self.args.get('keep_model_updated', True)

        self.mode = self.args.get('mode', 'optimal')
        if self.mode not in self.valid_modes:
            self._terminate_with_error(
                f'Invalid mode {self.mode}, must be one of {", ".join(self.valid_modes)}'
            )

        self.plot_interval = self.args.get('plot_interval', None)
        if self.plot_interval == 0:
            self.plot_interval = None

        setup_time_str = self.args.get('setup_time', self._read_setup_time())
        if setup_time_str is None:
            self.setup_time = None
        else:
            self.setup_time = self._parse_setup_time(setup_time_str)

        self.comfort_start_time = dt.time.fromisoformat(
            self.args['comfort_start_time'])
        self.comfort_end_time = dt.time.fromisoformat(
            self.args['comfort_end_time'])

        self.comfort_temperature = self.args['comfort_temperature']
        self.minimum_temperature = self.args['minimum_temperature']
        self.maximum_temperature = self.args.get('maximum_temperature',
                                                 self.comfort_temperature)
        if self.comfort_temperature < self.minimum_temperature:
            self._terminate_with_error(
                f'Minimum temperature ({self.minimum_temperature}) can not exceed comfort temperature ({self.comfort_temperature})'
            )
        if self.comfort_temperature > self.maximum_temperature:
            self._terminate_with_error(
                f'Comfort temperature ({self.comfort_temperature}) can not exceed maximum temperature ({self.maximum_temperature})'
            )

        self.thermostat = self.get_entity(self.args['thermostat_id'])

        if self.mode != 'classic':
            self.heating_power = self.args['heating_power']
            
            self.heater = self.get_entity(self.args['heater_id'])

            self.weather = self.get_entity(self.args['weather_id'])

            self.outside_thermometer = self.get_entity(
                self.args['outside_thermometer_id']
            ) if 'outside_thermometer_id' in self.args else None

            self.power_consumption_meter = self.get_entity(
                self.args['power_consumption_meter_id']
            ) if 'power_consumption_meter_id' in self.args else None

            self.power_price_sensor = self.get_entity(
                self.args.get('power_price_id',
                              None)) if 'power_price_id' in self.args else None

            self.tibber_access_token = self.args.get('tibber_access_token',
                                                     None)

    def delete_all_data(self):
        for folder in [
                self.data_path, self.figure_path, self.figure_data_path
        ]:
            delete_all_files_in_folder(folder)

    def start_running(self, *args):

        if self.mode == 'classic':
            self.begin_operation()
        else:
            if not self.has_model() and self.setup_time is None:
                self.gather_data_for_initial_model()
            else:
                self.update_models_and_begin_operation()

    def gather_data_for_initial_model(self, *args):
        self.log('Gathering data for initial model', level='INFO')

        self.setup_time = self._local_now()

        self.set_thermostat_temperature(self.minimum_temperature)

        self.run_in(self._wait_for_stable_temperature,
                    dt.timedelta(hours=1.0).total_seconds(),
                    heating_done=False,
                    max_temperature_rate=1.0,
                    min_temperature=0.95 * self.maximum_temperature,
                    min_remaining_heating_hours=2.0)

    def _wait_for_stable_temperature(self, kwargs):
        current_time = self._local_now()
        recorder = self._create_recorder_with_historical_data(
            self.setup_time, current_time, min_required_hours=1.0)
        if recorder is None:
            self.log('Not enough data to estimate temperature rate, waiting',
                     level='DEBUG')
            self.run_in(self._wait_for_stable_temperature,
                        dt.timedelta(hours=1).total_seconds(), **kwargs)
            return

        processed_record = recorder.create_processed_record()

        if np.abs(
                processed_record.temperature_rates[-1] *
                dt.timedelta(hours=1).total_seconds()
        ) <= kwargs['max_temperature_rate'] and processed_record.temperatures[
                -1] <= kwargs['min_temperature']:
            self.log('Temperature is stable', level='DEBUG')
            if kwargs.get('heating_done', True):
                self._complete_data_gathering()
            else:
                self.set_thermostat_temperature(self.maximum_temperature)
                kwargs['heating_start_time'] = self._local_now()
                self.run_in(self._wait_for_high_enough_temperature,
                            dt.timedelta(hours=0.5).total_seconds(), **kwargs)
        else:
            self.log('Temperature is not stable', level='DEBUG')
            self.run_in(self._wait_for_stable_temperature,
                        dt.timedelta(hours=1).total_seconds(), **kwargs)

    def _wait_for_high_enough_temperature(self, kwargs):
        if self.get_current_temperature() < kwargs['min_temperature']:
            self.log('Temperature is not high enough', level='DEBUG')
            self.run_in(self._wait_for_high_enough_temperature,
                        dt.timedelta(hours=0.5).total_seconds(), **kwargs)
        else:
            self.log('Temperature is high enough', level='DEBUG')
            heating_duration = self._local_now() - kwargs['heating_start_time']
            kwargs[
                'min_remaining_heating_hours'] -= heating_duration.total_seconds(
                ) / dt.timedelta(hours=1).total_seconds()
            if kwargs['min_remaining_heating_hours'] > 0.0:
                self.log('Total heating duration not long enough yet',
                         level='DEBUG')
            else:
                kwargs['heating_done'] = True
            self.set_thermostat_temperature(self.minimum_temperature)
            self._wait_for_stable_temperature(kwargs)

    def _complete_data_gathering(self):
        self._write_setup_time(self.setup_time)
        
        self.update_models(exclude_comfort_period=False)

        if self.control_heating:
            self.begin_operation()

    def update_models_and_begin_operation(self, **kwargs):
        self.update_models(**kwargs)
        self.begin_operation()
    
    def schedule_plot_data_updates(self):
        if self.has_model() or self.setup_time is not None or (not self.control_heating and not self.rebuild_model):
            self.run_in(self.create_plot_data, 20)

        self.run_every(self.create_plot_data,
                        f'now+{int(self.plot_interval)}',
                        self.plot_interval)
    
    def schedule_model_updates(self):
        if self.mode != 'classic':
            self.run_daily(self.update_models_callback,
                            self.comfort_start_time)

    def update_models_callback(self, kwargs):
        self.update_models(**kwargs)

    def update_models(self, **kwargs):
        self.log('Updating models with data from previous day', level='INFO')
        self.update_model_with_historical_data(**kwargs)
        self.update_forecast_correction_with_historical_data()

    def begin_operation(self):
        self.log('Beginning operation', level='INFO')
        if self._time_is_in_comfort_period():
            self._begin_comfort_period()
        else:
            self._perform_monitoring()

    def _perform_monitoring(self, *args):
        self.log('Performing monitoring', level='DEBUG')

        self.set_thermostat_temperature(self.minimum_temperature)

        if self.mode == 'classic':
            model = None
        else:
            model = self._read_model()
            if model is None:
                self.log(
                    f'Failed to read model file {self.model_path}, reverting to classic mode',
                    level='WARNING')

        current_datetime = self._local_now()
        current_time = datetime_to_timestamp(current_datetime)
        time_until_comfort_period = self.get_time_until_comfort_period(
            current_datetime=current_datetime)

        if model is None:
            self.log(
                f'Scheduling comfort period start in {time_until_comfort_period:g} s',
                level='DEBUG')
            self.run_in(self._begin_comfort_period, time_until_comfort_period)
        else:
            if self.heating_power is None:
                self._terminate_with_error('No heating power specified')

            self._update_temperature_forecast()

            times = self._calculate_modeling_times(
                current_time, current_time + time_until_comfort_period)

            current_temperature = self.get_current_temperature()
            outside_temperatures = self._obtain_projected_outside_temperatures(
                times, current_datetime)

            latest_required_heating_duration = model._compute_required_heating_duration_until_end_time(
                times, outside_temperatures, current_temperature,
                self.comfort_temperature, self.heating_power,
                current_time + time_until_comfort_period)

            if latest_required_heating_duration is None:
                self._transition_to_comfort_period_with_heating(
                    current_time, time_until_comfort_period)
            elif latest_required_heating_duration <= 0:
                self.log('Waiting for comfort period without heating',
                         level='DEBUG')
                self.run_in(self._begin_comfort_period,
                            time_until_comfort_period)
            else:
                time_until_latest_required_heating = time_until_comfort_period - latest_required_heating_duration

                time_until_next_monitoring = 0.5 * time_until_latest_required_heating

                self.scheduled_heating_period = (
                    current_time + time_until_latest_required_heating,
                    latest_required_heating_duration)

                if time_until_latest_required_heating <= self.modeling_buffer_time:
                    self._transition_to_comfort_period_with_heating(
                        current_time, time_until_comfort_period)
                elif self.mode == 'optimal':
                    self.log('Fetching power prices', level='DEBUG')
                    power_price = self._fetch_power_price()

                    if power_price.available:

                        def compute_duration(start_time):
                            duration = model._compute_required_heating_duration_from_start_time(
                                times, outside_temperatures,
                                current_temperature, self.comfort_temperature,
                                self.heating_power, start_time)

                            if duration is None:
                                duration = latest_required_heating_duration

                            return duration

                        def evaluate_price(start_time):
                            return power_price.compute_total_price_between(
                                start_time,
                                start_time + compute_duration(start_time))

                        result = scipy.optimize.minimize_scalar(
                            evaluate_price,
                            method='bounded',
                            bounds=(current_time, current_time +
                                    time_until_latest_required_heating))

                        if result.success:
                            best_start_time = result.x
                            best_heating_duration = compute_duration(
                                best_start_time)
                            time_until_best_start_time = best_start_time - current_time

                            self.log(
                                f'Optimal heating starts in {time_until_best_start_time:g} s, lasts {best_heating_duration:g} s and ends {(time_until_comfort_period - (time_until_best_start_time + best_heating_duration)):g} s before comfort period',
                                level='DEBUG')

                            self.scheduled_heating_period = (
                                best_start_time, best_heating_duration)

                            if time_until_best_start_time <= self.modeling_buffer_time:
                                self._schedule_preheating(
                                    current_time, time_until_best_start_time,
                                    best_heating_duration)
                                return
                            else:
                                time_until_next_monitoring = 0.5 * time_until_best_start_time
                        else:
                            self.log(
                                'Could not minimize heating price, reverting to smart mode',
                                level='WARNING')
                    else:
                        self.log(
                            'Power prices unavailable, reverting to smart mode',
                            level='WARNING')

                    self._schedule_monitoring(current_time,
                                              time_until_next_monitoring)

    def _calculate_modeling_times(self, start_time, end_time):
        duration = end_time - start_time
        n_times = int(np.ceil(duration / self.modeling_time_interval)) + 1
        times = np.linspace(start_time, end_time, n_times)
        return times

    def _obtain_projected_outside_temperatures(self,
                                               times,
                                               current_datetime,
                                               trim_times=False):
        forecasted_temperatures = self._evaluate_temperature_forecast(
            times, force_evaluation=trim_times)
        if trim_times:
            valid = times <= datetime_to_timestamp(
                self.temperature_forecast_end_time)
            times = times[valid]
            forecasted_temperatures = forecasted_temperatures[valid]
        outside_temperatures = self._adjust_forecasted_temperatures_to_fit_measurement(
            times, forecasted_temperatures, measured_time=current_datetime)
        if trim_times:
            return times, outside_temperatures
        else:
            return outside_temperatures

    def _fetch_power_price(self):
        if self.power_price is None:
            if self.tibber_access_token is None:
                self._terminate_with_error('No Tibber access token specified')
            self.power_price = TibberPowerPrice(self.tibber_access_token)
        self.power_price.fetch()
        return self.power_price

    def _schedule_preheating(self, current_time, time_until_heating_start,
                             heating_duration):
        self.log(
            f'Scheduling heating for {heating_duration:g} s in {time_until_heating_start:g} s',
            level='DEBUG')

        self.run_in(self._begin_preheating, time_until_heating_start)
        self.run_in(self._perform_monitoring,
                    time_until_heating_start + heating_duration)

        self.scheduled_heating_period = (current_time +
                                         time_until_heating_start,
                                         heating_duration)
        self.next_monitoring_time = current_time + time_until_heating_start + heating_duration

    def _schedule_monitoring(self,
                             current_time,
                             time_until_monitoring,
                             max_hours=0.5):
        time_until_monitoring = min(
            dt.timedelta(hours=max_hours).total_seconds(),
            time_until_monitoring)
        self.log(f'Scheduling new monitoring in {time_until_monitoring:g} s',
                 level='DEBUG')

        self.run_in(self._perform_monitoring, time_until_monitoring)

        self.next_monitoring_time = current_time + time_until_monitoring

    def _transition_to_comfort_period_with_heating(self, current_time,
                                                   time_until_comfort_period):
        self.log('Preheating for comfort period', level='DEBUG')

        self._begin_preheating()
        self.run_in(self._begin_comfort_period, time_until_comfort_period)

        self.scheduled_heating_period = (current_time,
                                         time_until_comfort_period)
        self.next_monitoring_time = None

    def _begin_preheating(self, *args):
        self.set_thermostat_temperature(self.comfort_temperature)

    def _stop_preheating(self, *args):
        self.set_thermostat_temperature(self.minimum_temperature)

    def _begin_comfort_period(self, *args):
        self.log('Beginning comfort period', level='DEBUG')

        self.set_thermostat_temperature(self.comfort_temperature)

        current_datetime = self._local_now()
        current_time = datetime_to_timestamp(current_datetime)
        time_until_monitoring = self.get_time_until_monitoring_period(
            current_datetime=current_datetime)

        self.run_in(self._perform_monitoring, time_until_monitoring)

        self.scheduled_heating_period = None
        self.next_monitoring_time = current_time + time_until_monitoring

    def _time_is_in_comfort_period(self, time=None):
        if time is None:
            time = self.time()
        if self.comfort_end_time >= self.comfort_start_time:
            return time >= self.comfort_start_time and time < self.comfort_end_time
        else:
            return time >= self.comfort_start_time or time < self.comfort_end_time

    def _datetimes_are_in_comfort_period(self, datetimes):
        return np.asarray([
            self._time_is_in_comfort_period(datetime.time())
            for datetime in datetimes
        ],
                          dtype=bool)

    def _find_comfort_period_transition_indices(self, datetimes):
        in_comfort_period = self._datetimes_are_in_comfort_period(datetimes)

        comfort_period_start_indices = list(
            np.nonzero(in_comfort_period[1:] > in_comfort_period[:-1])[0] + 1)
        if in_comfort_period[0]:
            comfort_period_start_indices.insert(0, 0)

        comfort_period_end_indices = list(
            np.nonzero(in_comfort_period[1:] < in_comfort_period[:-1])[0])
        if in_comfort_period[-1]:
            comfort_period_end_indices.append(in_comfort_period.size-1)

        return np.array(comfort_period_start_indices,
                        dtype=int), np.array(comfort_period_end_indices,
                                             dtype=int)

    def get_time_until_comfort_period(self, current_datetime=None):
        if current_datetime is None:
            current_datetime = self._local_now()
        return 0 if self._time_is_in_comfort_period(
            time=current_datetime.time()) else remaining_seconds_to_time(
                current_datetime, self.comfort_start_time)

    def get_time_until_monitoring_period(self, current_datetime=None):
        if current_datetime is None:
            current_datetime = self._local_now()
        return remaining_seconds_to_time(
            current_datetime,
            self.comfort_end_time) if self._time_is_in_comfort_period(
                time=current_datetime.time()) else 0

    def get_current_temperature(self):
        return float(
            self.thermostat.get_state(attribute='current_temperature'))

    def get_current_outside_temperature(self, allow_forecast_fallback=True):
        if self.outside_thermometer is None:
            if allow_forecast_fallback:
                self.log(
                    'No outside thermometer, using current forecasted temperature as current outside temperature',
                    level='DEBUG')
                return self.get_current_forecasted_temperature()
            else:
                self._terminate_with_error(
                    'No outside thermometer ID specified')
        else:
            return float(self.outside_thermometer.get_state())

    def get_current_forecasted_temperature(self):
        return float(self.weather.get_state(attribute='temperature'))

    def set_thermostat_temperature(self, temperature):
        self.log(f'Setting thermostat to {temperature}', level='DEBUG')
        assert temperature >= self.minimum_temperature
        self.call_service('climate/set_temperature',
                          entity_id=str(self.thermostat),
                          target_temp_low=self.minimum_temperature,
                          target_temp_high=temperature,
                          temperature=temperature,
                          hvac_mode='heat')
        self.thermostat.turn_on()

    def maintain_temperature_forecast(self):
        self.temperature_forecast, self.temperature_forecast_end_time = self._create_temperature_forecast_representation(
        )
        self.weather.listen_state(self._update_temperature_forecast_callback,
                                  attribute='forecast')

    def clear_measurement_table(self):
        if self.table_path.exists():
            self.log('Removing measurement table', level='INFO')
            os.remove(self.table_path)

    def update_model_with_historical_data(self,
                                          end_time=None,
                                          hours_back=24,
                                          min_required_hours=2.0,
                                          time_interval=600.0,
                                          update_heating_delay=True,
                                          exclude_comfort_period=True):
        self.log('Updating model with historical data', level='DEBUG')

        end_time = self._local_now() if end_time is None else end_time
        start_time = end_time - dt.timedelta(hours=hours_back)

        if self.setup_time is not None and start_time < self.setup_time:
            self.log('Increasing start time to setup time', level='DEBUG')
            start_time = self.setup_time

        recorder = self._create_recorder_with_historical_data(
            start_time,
            end_time,
            min_required_hours=min_required_hours,
            time_interval=time_interval,
            exclude_comfort_period=exclude_comfort_period)

        if recorder is None:
            self.log('Duration too short, aborting', level='DEBUG')
            return

        table = self._obtain_measurement_table()

        current_model = self._read_model()
        current_heating_delay = 0.0 if current_model is None else current_model.heating_delay

        processed_record = recorder.create_processed_record()
        new_heating_delay = processed_record.detect_heating_delay(
            fallback_value=current_heating_delay)

        processed_record.add_to_table(table)
        table.save()

        if table.fitting_possible:
            self.log('Fitting measurement table', level='DEBUG')
            new_model = table.create_model_from_fit(
                heating_delay=(new_heating_delay if update_heating_delay else
                               current_heating_delay),
                save_path=self.model_path)
            new_model.save()
        else:
            self.log('Not enough data to fit measurement table',
                     level='DEBUG')

    def clear_forecast_correction(self):
        if self.forecast_correction_path.exists():
            self.log('Removing forecast correction', level='INFO')
            os.remove(self.forecast_correction_path)

    def update_forecast_correction_with_historical_data(
            self,
            end_time=None,
            days_back=7.0,
            min_required_days=1.0,
            **kwargs):
        self.log('Updating forecast correction with historical data',
                 level='DEBUG')

        if self.outside_thermometer is None:
            self.log('No outside thermometer, aborting', level='DEBUG')
            return

        end_time = self._local_now() if end_time is None else end_time
        start_time = end_time - dt.timedelta(days=days_back)

        if self.setup_time is not None and start_time < self.setup_time:
            self.log('Increasing start time to setup time', level='DEBUG')
            start_time = self.setup_time

        if end_time < start_time + dt.timedelta(days=min_required_days):
            self.log('Duration too short, aborting', level='DEBUG')
            return

        forecasted_temperature_func = self._create_current_forecasted_temperature_representation(
            start_time=start_time, end_time=end_time)

        measured_temperature_func = self._create_outside_temperature_representation(
            allow_forecast_fallback=False,
            start_time=start_time,
            end_time=end_time)

        if forecasted_temperature_func is None or measured_temperature_func is None:
            self.log('Duration too short, aborting', level='DEBUG')
            return

        ForecastCorrection.from_historical_temperatures(
            forecasted_temperature_func,
            measured_temperature_func,
            start_time,
            end_time,
            save_path=self.forecast_correction_path,
            **kwargs).save()

    def create_plot_data(self, *args):
        if self.debug:
            self.set_log_level('INFO')

        self._update_temperature_forecast()

        current_time = self._local_now()
        self.plot_evolution(
            current_time - dt.timedelta(hours=48),
            current_time + dt.timedelta(hours=22),
            current_time,
            temperatures=['inside', 'outside', 'forecast'],
            indicators=[
                'current_time', 'comfort_period', 'heating',
                'comfort_temperature', 'minimum_temperature', 'monitoring_time'
            ],
            second_quantity='power_price',
            title='Evolution',
            data_save_path=(self.figure_data_path / 'evolution.pickle'))

        self.plot_measurement_scatter(
            hours_back=24,
            title='Measurements (last day)',
            data_save_path=(self.figure_data_path /
                            'measurements_1_day.pickle'))
        self.plot_measurement_scatter(
            hours_back=72,
            title='Measurements (last 3 days)',
            data_save_path=(self.figure_data_path /
                            'measurements_3_days.pickle'))

        self.plot_model_fit(title='Fitted model',
                            data_save_path=(self.figure_data_path /
                                            'model_fit.pickle'))

        # self.plot_forecast_observation_correlation(
        #     hours_back=7 * 24,
        #     data_save_path=(self.figure_data_path /
        #                     'forecast_correlation.png'))

        # self.compare_modeled_to_actual_temperature_evolution(
        #     self._local_now() - dt.timedelta(hours=22), hours_duration=22.0)

        if self.debug:
            self.set_log_level('DEBUG')

    def plot_measurement_scatter(self,
                                 end_time=None,
                                 hours_back=24.0,
                                 **kwargs):
        self.log('Plotting measurement scatter', level='DEBUG')

        end_time = self._local_now() if end_time is None else end_time
        start_time = end_time - dt.timedelta(hours=hours_back)

        if self.setup_time is not None and start_time < self.setup_time:
            self.log('Increasing start time to setup time', level='DEBUG')
            start_time = self.setup_time

        recorder = self._create_recorder_with_historical_data(
            start_time, end_time, exclude_comfort_period=True)

        if recorder is None:
            self.log('Duration too short, aborting', level='DEBUG')
            return

        processed_record = recorder.create_processed_record()

        processed_record.plot_scatter(**kwargs)

    def plot_evolution(self,
                       start_time,
                       end_time,
                       current_time,
                       temperatures=[],
                       indicators=[],
                       second_quantity=None,
                       **kwargs):

        artists = []
        second_ylabel = None

        past_datetimes = []
        future_datetimes = []

        if start_time < current_time:

            if self.setup_time is not None and start_time < self.setup_time:
                self.log('Increasing start time to setup time',
                         level='DEBUG')
                start_time = self.setup_time

            recorder = self._create_recorder_with_historical_data(
                start_time, current_time)

            if recorder is None:
                self.log('Past duration too short, skipping historical data',
                         level='DEBUG')
            else:
                processed_record = recorder.create_processed_record()
                past_datetimes = self.timestamps_to_local_datetimes(
                    processed_record.times)

                if 'heating' in indicators:
                    artists.append(
                        dict(type='fill',
                             x_values=past_datetimes,
                             y_values=processed_record.heating_powers,
                             y_threshold=50.0,
                             color='tab:red',
                             alpha=0.5))

                if 'inside' in temperatures:
                    artists.append(
                        dict(x_values=past_datetimes,
                             y_values=processed_record.temperatures,
                             color='tab:green',
                             label='Temp. inside'))

                if 'outside' in temperatures:
                    artists.append(
                        dict(x_values=past_datetimes,
                             y_values=processed_record.outside_temperatures,
                             color='tab:blue',
                             label='Temp. outside'))

                if 'forecast' in temperatures:
                    forecasted_temperature_func = self._create_current_forecasted_temperature_representation(
                        start_time=start_time, end_time=current_time)

                    if forecasted_temperature_func is None:
                        self.log(
                            'Past duration too short, skipping historical forecast',
                            level='DEBUG')
                    else:
                        correction = self._obtain_forecast_correction()
                        corrected_forecasted_temperature_func = correction.corrected(
                            forecasted_temperature_func)

                        artists.append(
                            dict(
                                x_values=past_datetimes,
                                y_values=corrected_forecasted_temperature_func(
                                    processed_record.times),
                                color='tab:cyan',
                                label='Temp. forecast'))

                if second_quantity == 'heating_power':
                    artists.append(
                        dict(ax=1,
                             x_values=past_datetimes,
                             y_values=processed_record.heating_powers,
                             color='tab:red',
                             label='Heating'))
                    second_ylabel = 'Heating power [W]'
                elif second_quantity == 'temperature_rate':
                    artists.append(
                        dict(ax=1,
                             x_values=past_datetimes,
                             y_values=processed_record.temperature_rates *
                             dt.timedelta(hours=1).total_seconds(),
                             color='tab:purple',
                             label='Temp. change'))
                    second_ylabel = 'Rate of change [°C/hour]'
                elif second_quantity == 'power_price':
                    power_price = self._create_power_price_representation(
                        start_time=start_time, end_time=current_time)
                    if power_price is None:
                        power_price = self._fetch_power_price()
                        if power_price.available:
                            earliest_time = datetime_to_timestamp(
                                power_price.earliest_time)
                            valid_times = processed_record.times[
                                processed_record.times >= earliest_time]
                            artists.append(
                                dict(
                                    ax=1,
                                    x_values=self.
                                    timestamps_to_local_datetimes(valid_times),
                                    y_values=power_price(valid_times),
                                    color='tab:purple',
                                    label='Power price'))
                            second_ylabel = 'Price [NOK/kWh]'
                        else:
                            self.log('Power prices unavailable, skipping',
                                     level='DEBUG')
                    else:
                        artists.append(
                            dict(ax=1,
                                 x_values=past_datetimes,
                                 y_values=power_price(processed_record.times),
                                 color='tab:purple',
                                 label='Power price'))
                        second_ylabel = 'Price [NOK/kWh]'

        if 'current_time' in indicators:
            artists.append(
                dict(type='vline',
                     x_value=current_time,
                     ls='-',
                     color='tab:gray'))

        if 'comfort_temperature' in indicators:
            artists.append(
                dict(type='hline',
                     y_value=self.comfort_temperature,
                     ls='--',
                     color='darkorange'))

        if 'minimum_temperature' in indicators:
            artists.append(
                dict(type='hline',
                     y_value=self.minimum_temperature,
                     ls='--',
                     color='mediumblue'))

        if end_time > current_time:
            future_times = self._calculate_modeling_times(
                datetime_to_timestamp(current_time),
                datetime_to_timestamp(end_time))

            future_datetimes = self.timestamps_to_local_datetimes(future_times)

        all_datetimes = past_datetimes + future_datetimes

        if 'comfort_period' in indicators:
            artists.append(
                dict(type='fill',
                     x_values=all_datetimes,
                     y_values=self._datetimes_are_in_comfort_period(
                         all_datetimes),
                     y_threshold=0.5,
                     color='darkgreen',
                     alpha=0.2))

        if end_time > current_time:
            outside_temperatures = self._obtain_projected_outside_temperatures(
                future_times, current_time)

            if 'forecast' in temperatures:
                artists.append(
                    dict(x_values=future_datetimes,
                         y_values=outside_temperatures,
                         color='tab:cyan',
                         label='Temp. forecast'))

            if second_quantity == 'power_price':
                power_price = self._fetch_power_price()
                if power_price.available:
                    latest_time = datetime_to_timestamp(
                        power_price.latest_time)
                    valid_times = future_times[future_times <= latest_time]
                    artists.append(
                        dict(ax=1,
                             x_values=self.timestamps_to_local_datetimes(
                                 valid_times),
                             y_values=power_price(valid_times),
                             color='tab:purple',
                             label='Power price'))
                    second_ylabel = 'Price [NOK/kWh]'
                else:
                    self.log('Power prices unavailable, skipping',
                             level='DEBUG')

            model = self._read_model()

            if model is None:
                self.log('No model found, skipping projected data',
                         level='DEBUG')
            else:
                initial_temperature = self.get_current_temperature()

                if self.scheduled_heating_period is None:
                    max_heating_power, heating_start_time, heating_end_time = None, None, None
                    planned_heating_powers = np.zeros_like(future_times)
                else:
                    max_heating_power = np.full(1, self.heating_power)
                    heating_start_time = np.full(
                        1, self.scheduled_heating_period[0])
                    heating_end_time = np.full(
                        1, sum(self.scheduled_heating_period))
                    planned_heating_powers = self.heating_power * np.logical_and(
                        future_times >= self.scheduled_heating_period[0],
                        future_times < sum(self.scheduled_heating_period))

                if self.control_heating:
                    modeled_temperatures = model.compute_evolution_with_thermostat(
                        future_times,
                        outside_temperatures,
                        initial_temperature,
                        self.comfort_temperature,
                        self.minimum_temperature,
                        *self._find_comfort_period_transition_indices(
                            future_datetimes),
                        heating_powers=max_heating_power,
                        heating_start_times=heating_start_time,
                        heating_end_times=heating_end_time, logger=self.log)
                else:
                    modeled_temperatures = model.compute_evolution(
                        future_times,
                        outside_temperatures,
                        initial_temperature,
                        heating_powers=max_heating_power,
                        heating_start_times=heating_start_time,
                        heating_end_times=heating_end_time)

                if 'heating' in indicators and self.scheduled_heating_period is not None:
                    artists.append(
                        dict(type='fill',
                             x_values=future_datetimes,
                             y_values=planned_heating_powers,
                             y_threshold=50.0,
                             color='tab:red',
                             alpha=0.5))

                if 'monitoring_time' in indicators and self.next_monitoring_time is not None:
                    artists.append(
                        dict(type='vline',
                             x_value=self.timestamp_to_local_datetime(
                                 self.next_monitoring_time),
                             ls=':',
                             color='tab:olive'))

                if 'inside' in temperatures:
                    artists.append(
                        dict(x_values=future_datetimes,
                             y_values=modeled_temperatures,
                             color='tab:green',
                             label='Temp. inside'))

                if second_quantity == 'heating_power':
                    artists.append(
                        dict(ax=1,
                             x_values=future_datetimes,
                             y_values=planned_heating_powers,
                             color='tab:red',
                             label='Heating'))
                    second_ylabel = 'Heating power [W]'
                elif second_quantity == 'temperature_rate':
                    temperature_rates = np.gradient(modeled_temperatures,
                                                    future_times)
                    artists.append(
                        dict(ax=1,
                             x_values=future_datetimes,
                             y_values=temperature_rates *
                             dt.timedelta(hours=1).total_seconds(),
                             color='tab:purple',
                             label='Temp. change'))
                    second_ylabel = 'Rate of change [°C/hour]'

        plot(*artists,
             x_uses_datetimes=True,
             xlim=(start_time, end_time),
             ylabel='Temperature [°C]',
             second_ylabel=second_ylabel,
             **kwargs)

    def plot_model_fit(self, **kwargs):
        self.log('Plotting model fit', level='DEBUG')
        if self.table_path.exists():
            table = self._obtain_measurement_table()
            table.plot(**kwargs)
        else:
            self.log('No measurements found, aborting', level='DEBUG')

    def plot_temperature_forecast(self, start_time=None, hours_ahead=24.0):
        self.log('Plotting temperature forecast', level='DEBUG')
        start_time = self._local_now() if start_time is None else start_time
        end_time = start_time + dt.timedelta(hours=hours_ahead)

        if self.setup_time is not None and start_time < self.setup_time:
            self.log('Start time earlier than setup time, aborting',
                     level='DEBUG')
            return

        times = np.linspace(datetime_to_timestamp(start_time),
                            datetime_to_timestamp(end_time), 3600)

        plot(dict(x_values=self.timestamps_to_local_datetimes(times),
                  y_values=self._evaluate_temperature_forecast(times),
                  label='Forecast'),
             dict(x_values=[self._local_now()],
                  y_values=[self.get_current_outside_temperature()],
                  marker='o',
                  ls='',
                  label='Current'),
             x_uses_datetimes=True,
             ylabel='Temperature [°C]',
             output_path=(self.figure_path / 'forecast.png'))

    def plot_forecast_observation_correlation(self,
                                              end_time=None,
                                              hours_back=24.0,
                                              time_interval=600.0,
                                              **kwargs):
        self.log(
            'Plotting correlation between forecasted and observed temperatures',
            level='DEBUG')

        if self.outside_thermometer is None:
            self.log('No outside thermometer, aborting', level='DEBUG')
            return

        end_time = self._local_now() if end_time is None else end_time
        start_time = end_time - dt.timedelta(hours=hours_back)

        if self.setup_time is not None and start_time < self.setup_time:
            self.log('Increasing start time to setup time', level='DEBUG')
            start_time = self.setup_time

        forecasted_temperature_func = self._create_current_forecasted_temperature_representation(
            start_time=start_time, end_time=end_time)

        measured_temperature_func = self._create_outside_temperature_representation(
            allow_forecast_fallback=False,
            start_time=start_time,
            end_time=end_time)

        if forecasted_temperature_func is None or measured_temperature_func is None:
            self.log('Duration too short, aborting', level='DEBUG')
            return

        start_timestamp = datetime_to_timestamp(start_time)
        end_timestamp = datetime_to_timestamp(end_time)
        n_times = int((end_timestamp - start_timestamp) / time_interval) + 1
        times = np.linspace(start_timestamp, end_timestamp, n_times)

        forecasted_temperatures = forecasted_temperature_func(times)
        measured_temperatures = measured_temperature_func(times)

        min_temp = min(forecasted_temperatures.min(),
                       measured_temperatures.min())
        max_temp = max(forecasted_temperatures.max(),
                       measured_temperatures.max())
        max_abs_temp = max(abs(min_temp), abs(max_temp))

        correction = self._obtain_forecast_correction()

        plot(dict(x_values=[-max_abs_temp, max_abs_temp],
                  y_values=[-max_abs_temp, max_abs_temp],
                  color='tab:gray',
                  ls='--'),
             dict(x_values=[-max_abs_temp, max_abs_temp],
                  y_values=[
                      correction(-max_abs_temp),
                      correction(max_abs_temp)
                  ],
                  color='tab:cyan',
                  ls='--'),
             dict(type='scatter',
                  x_values=forecasted_temperature_func(times),
                  y_values=measured_temperature_func(times),
                  color='tab:orange',
                  s=10),
             xlabel='Forecasted temperature [°C]',
             ylabel='Measured temperature [°C]',
             aspect='equal',
             **kwargs)

    def compare_modeled_to_actual_temperature_evolution(
            self,
            start_time,
            hours_duration=16.0,
            time_interval=300.0,
            **kwargs):
        self.log('Comparing modeled to actual temperature evolution',
                 level='DEBUG')

        duration = hours_duration * dt.timedelta(hours=1).total_seconds()

        if self.setup_time is not None and start_time < self.setup_time:
            self.log('Start time earlier than setup time, aborting',
                     level='DEBUG')
            return

        end_time = start_time + dt.timedelta(seconds=duration)

        recorder = self._create_recorder_with_historical_data(
            start_time, end_time, time_interval=time_interval, **kwargs)

        if recorder is None:
            self.log('Not enough data, aborting', level='DEBUG')
            return

        n_times = int(np.ceil(duration / time_interval)) + 1
        times = datetime_to_timestamp(
            start_time) + np.arange(n_times) * time_interval

        processed_record = recorder.create_processed_record()

        result = self._create_historical_temperature_forecast_representation(
            start_time)

        if result is None:
            self.log('Not enough data, aborting', level='DEBUG')
            return

        temperature_forecast, temperature_forecast_end_time = result

        forecasted_temperatures = self._evaluate_temperature_forecast(
            times,
            forecast=temperature_forecast,
            forecast_end_time=temperature_forecast_end_time,
            allow_thermometer_fallback=True)

        temperatures = processed_record.temperatures
        outside_temperatures = processed_record.outside_temperatures
        heating_powers = processed_record.heating_powers

        result = processed_record.find_heating_periods()
        if result is None:
            mean_heating_powers, heating_start_times, heating_end_times = None, None, None
        else:
            mean_heating_powers, heating_start_times, heating_end_times = result

        model = self._read_model()

        initial_temperature = temperatures[0]
        initial_outside_temperature = outside_temperatures[0]
        forecasted_temperatures = self._adjust_forecasted_temperatures_to_fit_measurement(
            times,
            forecasted_temperatures,
            measured_time=start_time,
            measured_temperature=initial_outside_temperature)

        modeled_temperatures = model.compute_evolution(
            times,
            outside_temperatures,
            initial_temperature,
            heating_powers=mean_heating_powers,
            heating_start_times=heating_start_times,
            heating_end_times=heating_end_times)

        datetimes = self.timestamps_to_local_datetimes(times)
        plot(dict(x_values=datetimes,
                  y_values=modeled_temperatures,
                  label='Inside (modeled)'),
             dict(x_values=datetimes,
                  y_values=temperatures,
                  label='Inside (measured)'),
             dict(x_values=datetimes,
                  y_values=forecasted_temperatures,
                  label='Outside (forecasted)'),
             dict(x_values=datetimes,
                  y_values=outside_temperatures,
                  label='Outside (measured)'),
             dict(type='fill',
                  x_values=datetimes,
                  y_values=heating_powers,
                  y_threshold=50.0,
                  color='tab:red',
                  alpha=0.2),
             x_uses_datetimes=True,
             output_path=(self.figure_path /
                          'modeled_temperature_evolution.png'))

    def _terminate_with_error(self, message):
        self.log(message, level='CRITICAL')
        self.stop_app(self.name)
        raise RuntimeError('Terminated')

    def _localize_datetime(self, datetime, strip_tzinfo=True):
        local_datetime = datetime.astimezone(self._tz)
        return local_datetime.replace(
            tzinfo=None) if strip_tzinfo else local_datetime

    def _local_now(self):
        return self._localize_datetime(self.datetime(), strip_tzinfo=False)

    def timestamp_to_local_datetime(self, timestamp, strip_tzinfo=False):
        return self._localize_datetime(timestamp_to_datetime(timestamp),
                                       strip_tzinfo=strip_tzinfo)

    def timestamps_to_local_datetimes(self, timestamps, **kwargs):
        return [
            self.timestamp_to_local_datetime(timestamp, **kwargs)
            for timestamp in timestamps
        ]

    def _create_recorder_with_historical_data(self,
                                              start_time,
                                              end_time,
                                              min_required_hours=2.0,
                                              time_interval=300.0,
                                              exclude_comfort_period=False):
        self.log('Creating recorder with historical data', level='DEBUG')

        if end_time < start_time + dt.timedelta(hours=min_required_hours):
            self.log('Too short duration for historical data', level='DEBUG')
            return None

        start_timestamp = datetime_to_timestamp(start_time)
        end_timestamp = datetime_to_timestamp(end_time)
        n_times = int((end_timestamp - start_timestamp) / time_interval) + 1
        times = np.linspace(start_timestamp, end_timestamp, n_times)

        temperature_func = self._create_temperature_representation(
            start_time=start_time, end_time=end_time)

        outside_temperature_func = self._create_outside_temperature_representation(
            allow_forecast_fallback=True,
            start_time=start_time,
            end_time=end_time)

        power_consumption_func = self._create_power_consumption_representation(
            start_time=start_time, end_time=end_time)

        if temperature_func is None or outside_temperature_func is None or power_consumption_func is None:
            self.log('Too short duration for historical data', level='DEBUG')
            return None

        recorder = TemperatureRecorder()

        datetimes = self.timestamps_to_local_datetimes(times)

        if exclude_comfort_period:
            datetimes = list(
                filter(
                    lambda datetime: not self._time_is_in_comfort_period(
                        datetime.time()), datetimes))
            times = datetimes_to_timestamps(datetimes)

        heating_powers = power_consumption_func(times)
        temperatures = temperature_func(times)
        outside_temperatures = outside_temperature_func(times)

        recorder.add_measurements(datetimes, heating_powers, temperatures,
                                  outside_temperatures)

        return recorder

    def _evaluate_temperature_forecast(self,
                                       times,
                                       forecast=None,
                                       forecast_end_time=None,
                                       allow_thermometer_fallback=True,
                                       force_evaluation=False):
        if forecast is None or forecast_end_time is None:
            forecast = self.temperature_forecast
            forecast_end_time = self.temperature_forecast_end_time

        if forecast is None or forecast_end_time is None or (
                not force_evaluation
                and datetime_to_timestamp(forecast_end_time) < times[-1]):
            if allow_thermometer_fallback:
                self.log('Using temperatures from previous day as forecast',
                         level='DEBUG')
                start_time = self.timestamp_to_local_datetime(times[0])
                previous_day_start_time = start_time - dt.timedelta(days=1)
                previous_day_times = times - dt.timedelta(
                    days=1).total_seconds()

                measured_temperature_func = self._create_outside_temperature_representation(
                    allow_forecast_fallback=True,
                    start_time=previous_day_start_time,
                    end_time=start_time)

                if measured_temperature_func is None:
                    self._terminate_with_error('Failed to obtain forecast')

                forecasted_temperatures = measured_temperature_func(
                    previous_day_times) + (
                        measured_temperature_func(times[0]) -
                        measured_temperature_func(previous_day_times[0]))
            else:
                self._terminate_with_error('Failed to obtain forecast')
        else:
            self.log('Evaluating forecast', level='DEBUG')
            forecasted_temperatures = forecast(times)

        return forecasted_temperatures

    def _create_sensor_state_representation(self,
                                            sensor_id,
                                            smoothing_window_duration=3000.0,
                                            **kwargs):
        times, values = self._fetch_historical_state(sensor_id, **kwargs)

        if len(times) < 4:
            return None

        if smoothing_window_duration is not None:
            values = ProcessedTemperatureRecord.compute_smooth_evolution(
                datetimes_to_timestamps(times), np.asfarray(values),
                smoothing_window_duration)

        return scipy.interpolate.UnivariateSpline(
            datetimes_to_timestamps(times), values, s=0, ext='const')

    def _create_temperature_representation(self, **kwargs):
        self.log('Creating temperature representation', level='DEBUG')
        return self._create_sensor_state_representation(
            str(self.thermostat),
            extractor=lambda entry: entry['attributes']['current_temperature'],
            **kwargs)

    def _create_outside_temperature_representation(
            self, allow_forecast_fallback=True, **kwargs):
        self.log('Creating outside temperature representation', level='DEBUG')
        if self.outside_thermometer is None:
            if allow_forecast_fallback:
                self.log(
                    'No outside thermometer, using forecasted temperatures as outside temperatures',
                    level='DEBUG')
                return self._create_current_forecasted_temperature_representation(
                    **kwargs)
            else:
                self._terminate_with_error(
                    'No outside thermometer ID specified')
        else:
            return self._create_sensor_state_representation(
                str(self.outside_thermometer), **kwargs)

    def _create_power_consumption_representation(self, **kwargs):
        self.log('Creating power consumption representation', level='DEBUG')
        if self.power_consumption_meter is None:
            self.log(
                'No power consumption meter, using constant heating power',
                level='DEBUG')
            if self.heating_power is None:
                self._terminate_with_error('No heating power specified')
            times, values = self._fetch_historical_state(
                str(self.heater),
                extractor=lambda entry: entry['state'] == 'on',
                value_type=int,
                **kwargs)
            values = np.asarray(values) * self.heating_power
            interp_kind = 'previous'
        else:
            if self.power_consumption_meter is None:
                self._terminate_with_error(
                    'No power consumption meter ID specified')
            times, values = self._fetch_historical_state(
                str(self.power_consumption_meter), **kwargs)
            interp_kind = 'nearest'

        if len(times) < 2:
            return None

        return scipy.interpolate.interp1d(datetimes_to_timestamps(times),
                                          values,
                                          kind=interp_kind,
                                          copy=False,
                                          fill_value='extrapolate',
                                          assume_sorted=True)

    def _create_power_price_representation(self, **kwargs):
        self.log('Creating power price representation', level='DEBUG')
        if self.power_price_sensor is None:
            return None
        else:
            return self._create_sensor_state_representation(
                str(self.power_price_sensor),
                smoothing_window_duration=None,
                **kwargs)

    def _create_current_forecasted_temperature_representation(self, **kwargs):
        self.log('Creating forecasted temperature representation',
                 level='DEBUG')
        return self._create_sensor_state_representation(
            str(self.weather),
            smoothing_window_duration=None,
            extractor=lambda entry: entry['attributes']['temperature'],
            **kwargs)

    def _update_temperature_forecast(self, **kwargs):
        self.temperature_forecast, self.temperature_forecast_end_time = self._create_temperature_forecast_representation(
            **kwargs)

    def _update_temperature_forecast_callback(self, entity, attribute, old,
                                              new, kwargs):
        self._update_temperature_forecast(forecast=new)

    def _create_historical_temperature_forecast_representation(
            self, time, max_hours_back=25.0):
        time = self._localize_datetime(time)
        forecast = None
        delta = dt.timedelta(hours=0.5)
        while forecast is None and delta < dt.timedelta(hours=max_hours_back):
            history = self.get_history(entity_id=str(self.weather),
                                       start_time=(time - delta),
                                       end_time=time)[0]
            for entry in history[::-1]:
                if 'forecast' in entry['attributes']:
                    forecast = entry['attributes']['forecast']
            delta *= 2

        if forecast is None:
            return None

        return self._create_temperature_forecast_representation(
            forecast=forecast)

    def _create_temperature_forecast_representation(self,
                                                    forecast=None,
                                                    corrected=True):
        self.log('Creating temperature forecast representation', level='DEBUG')
        if forecast is None:
            forecast = self.weather.get_state(attribute='forecast')

        times, temperatures = zip(*((self.convert_utc(entry['datetime']),
                                     entry['temperature'])
                                    for entry in forecast))

        temperature_forecast_end_time = times[-1]

        timestamps = datetimes_to_timestamps(times)

        temperature_func = scipy.interpolate.UnivariateSpline(timestamps,
                                                              temperatures,
                                                              s=0)

        if corrected:
            correction = self._obtain_forecast_correction()
            temperature_func = correction.corrected(temperature_func)

        return temperature_func, temperature_forecast_end_time

    def _adjust_forecasted_temperatures_to_fit_measurement(
            self,
            times,
            forecasted_temperatures,
            measured_time=None,
            measured_temperature=None,
            **kwargs):
        self.log('Adjusting forecasted temperatures', level='DEBUG')
        if measured_time is None:
            measured_time = self._local_now()
        if measured_temperature is None:
            measured_temperature = self.get_current_outside_temperature()
        return ForecastCorrection.adjust_forecast_to_fit_measured_temperature(
            times, forecasted_temperatures,
            datetime_to_timestamp(measured_time), measured_temperature,
            **kwargs)

    def _fetch_historical_state(self,
                                entity_id,
                                start_time=None,
                                end_time=None,
                                hours_back=24.0,
                                extractor=lambda entry: entry['state'],
                                value_type=float):
        end_time = self.datetime() if end_time is None else end_time
        start_time = (end_time - dt.timedelta(hours=hours_back)
                      ) if start_time is None else start_time

        history = self.get_history(
            entity_id=entity_id,
            start_time=self._localize_datetime(start_time),
            end_time=self._localize_datetime(end_time))

        times = []
        values = []

        if history is not None and len(history) > 0:
            history = history[0]

            for entry in history:
                if entry['state'] not in ('unavailable', 'unknown', ''):
                    value = extractor(entry)
                    if value is not None:
                        times.append(self.convert_utc(entry['last_updated']))
                        values.append(value_type(value))

        return times, values

    def _parse_setup_time(self, setup_time_str):
        setup_time = dt.datetime.fromisoformat(setup_time_str)
        if setup_time.tzinfo is None:
            setup_time = setup_time.replace(tzinfo=self._tz)
        return setup_time

    def _read_setup_time(self):
        if self.setup_time_path.exists():
            with open(self.setup_time_path, 'r') as f:
                return f.read()
        else:
            return None

    def _write_setup_time(self, setup_time):
        with open(self.setup_time_path, 'w') as f:
            f.write(setup_time.isoformat())

    def _obtain_measurement_table(self):
        if self.table_path.exists():
            return MeasurementTable.from_file(self.table_path)
        else:
            return MeasurementTable.new(save_path=self.table_path)

    def has_model(self):
        return self.model_path.exists()

    def _read_model(self):
        if self.model_path.exists():
            return NewtonHeatingModel.from_file(self.model_path)
        else:
            return None

    def _obtain_forecast_correction(self):
        if self.forecast_correction_path.exists():
            return ForecastCorrection.from_file(self.forecast_correction_path)
        else:
            return ForecastCorrection(save_path=self.forecast_correction_path)


class ThermostatSetting:

    @staticmethod
    def from_temperature_array(start_datetime: dt.datetime, interval: dt.timedelta, temperatures: np.ndarray, cyclic: bool = False) -> 'ThermostatSetting':
        assert temperatures.size > 0
        
        if temperatures.size == 1:
            transition_times = np.full(1, interval.total_seconds())
            temperature_settings = temperatures
        else:
            next_transition_indices = np.nonzero(temperatures[1:] != temperatures[:-1]) + 1
            transition_indices = np.zeros(next_transition_indices.size + 1, dtype=next_transition_indices.dtype)
            transition_indices[1:] = next_transition_indices
            transition_times = interval.total_seconds()*transition_indices
            temperature_settings = temperatures[transition_indices]
        
        return ThermostatSetting(start_datetime, transition_times, temperature_settings)

    def __init__(self, start_datetime: dt.datetime, transition_times: np.ndarray, temperature_settings: np.ndarray, cyclic: bool = False) -> None:
        self._start_datetime = start_datetime

        self._transition_times = transition_times.copy()
        self._temperature_settings = temperature_settings.copy()
        
        self._temperature_func = scipy.interpolate.interp1d(self._transition_times, self._temperature_settings, kind='previous', fill_value='extrapolate', assume_sorted=True, copy=False)

    def _datetime_to_relative_time(self, datetime: dt.datetime) -> float:
        return (datetime - self._start_datetime).total_seconds()

    def _relative_time_to_datetime(self, time: float) -> dt.datetime:
        return self._start_datetime + dt.timedelta(seconds=time)

    def _get_next_transition_idx(self, time: float, inclusive: bool = True) -> int:
        return np.searchsorted(self._transition_times, time, side=('left' if inclusive else 'right'))

    def _get_previous_transition_idx(self, time: float, inclusive: bool = True) -> int:
        return np.searchsorted(self._transition_times, time, side=('right' if inclusive else 'left')) - 1

    def get_temperature(self, datetime: dt.datetime) -> float:
        return self._temperature_func(self._datetime_to_relative_time(datetime))

    def get_next_transition_time(self, datetime: dt.datetime, inclusive: bool = True, return_temperature: bool = False) -> Union[dt.datetime, Tuple[dt.datetime, float]]:
        time = self._datetime_to_relative_time(datetime)
        next_time_idx = self._get_next_transition_idx(time, inclusive=inclusive)
        next_transition_datetime = self._relative_time_to_datetime(self._transition_times[next_time_idx])
        if return_temperature:
            return next_transition_datetime, self._temperature_settings[next_time_idx]
        else:
            return next_transition_datetime

    def get_next_transition(self, datetime: dt.datetime) -> Tuple[dt.datetime, float]:
        return self.get_next_transition_time(datetime, return_temperature=True)

    def for_subinterval(self, min_start_datetime: dt.datetime, max_end_datetime: dt.datetime) -> 'ThermostatSetting':
        min_start_time = self._datetime_to_relative_time(min_start_datetime)
        new_start_idx = self._get_next_transition_idx(min_start_time, inclusive=True)
        new_start_datetime = self._relative_time_to_datetime(self._transition_times[new_start_idx])

        max_end_time = self._datetime_to_relative_time(max_end_datetime)
        new_end_idx = self._get_previous_transition_idx(max_end_time, inclusive=True)
        
        subinterval_slice = slice(new_start_idx, new_end_idx + 1)
        return ThermostatSetting(new_start_datetime, self._transition_times[subinterval_slice], self._temperature_settings[subinterval_slice])

    def __iter__(self) -> Iterable:
        return ((self._relative_time_to_datetime(time), temperature) for time, temperature in zip(self._transition_times, self._temperature_settings))


class ForecastCorrection:

    @staticmethod
    def from_historical_temperatures(forecasted_temperature_func: Callable,
                                     measured_temperature_func: Callable,
                                     start_time: dt.datetime,
                                     end_time: dt.datetime,
                                     time_interval: float = 9000.0,
                                     max_difference: float = 7.0,
                                     min_valid_points: int = 10,
                                     **kwargs) -> 'ForecastCorrection':
        start_timestamp = datetime_to_timestamp(start_time)
        end_timestamp = datetime_to_timestamp(end_time)
        n_times = int((end_timestamp - start_timestamp) / time_interval) + 1
        times = np.linspace(start_timestamp, end_timestamp, n_times)

        forecasted_temperatures = forecasted_temperature_func(times)
        measured_temperatures = measured_temperature_func(times)

        temperature_differences = np.abs(measured_temperatures -
                                         forecasted_temperatures)
        valid_mask = temperature_differences <= max_difference

        if np.sum(valid_mask) < min_valid_points:
            return ForecastCorrection(**kwargs)

        valid_forecasted_temperatures = forecasted_temperatures[valid_mask]
        valid_measured_temperatures = measured_temperatures[valid_mask]

        regression = sklearn.linear_model.LinearRegression()
        regression.fit(valid_forecasted_temperatures[:, np.newaxis],
                       valid_measured_temperatures)

        return ForecastCorrection(regression.intercept_, regression.coef_[0],
                                  **kwargs)

    @staticmethod
    def adjust_forecast_to_fit_measured_temperature(
            forecasted_times: np.ndarray,
            forecasted_temperatures: np.ndarray,
            measured_time: float,
            measured_temperature: float,
            influence_time: float = 18000.0,
            localness: float = 1.0) -> np.ndarray:
        time_diffs = np.abs(forecasted_times - measured_time)
        correction_factors = np.maximum(
            0, 1.0 - (time_diffs / influence_time)**localness)
        temperature_at_measured_time = np.interp(measured_time,
                                                 forecasted_times,
                                                 forecasted_temperatures)
        corrected_temperatures = forecasted_temperatures + (
            measured_temperature -
            temperature_at_measured_time) * correction_factors
        return corrected_temperatures

    @staticmethod
    def from_file(save_path: Union[str, Path]) -> 'ForecastCorrection':
        data = np.load(save_path)
        return ForecastCorrection(data['intercept'].item(),
                                  data['slope'].item(),
                                  save_path=save_path)

    def __init__(self,
                 intercept: float = 0.0,
                 slope: float = 1.0,
                 save_path: Optional[Union[str, Path]] = None) -> None:
        self._intercept = intercept
        self._slope = slope
        self.save_path = None if save_path is None else Path(save_path)

    def save(self):
        if self.save_path is not None:
            np.savez(self.save_path,
                     intercept=self._intercept,
                     slope=self._slope)

    def __call__(
        self, forecasted_temperature: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        return self._intercept + self._slope * forecasted_temperature

    def corrected(self, forecasted_temperature_func: Callable) -> Callable:
        return lambda arg: self._intercept + self._slope * forecasted_temperature_func(
            arg)


class PowerPrice:

    def __init__(self) -> None:
        self._price_func = None
        self._earliest_time = None
        self._latest_time = None
        self._unit = None

    @property
    def available(self) -> bool:
        return self._price_func is not None

    @property
    def earliest_time(self) -> Union[dt.datetime, None]:
        return self._earliest_time

    @property
    def latest_time(self) -> Union[dt.datetime, None]:
        return self._latest_time

    @property
    def unit(self) -> Union[str, None]:
        return self._unit

    def fetch(self) -> None:
        raise NotImplementedError

    def __call__(self, times: np.ndarray) -> np.ndarray:
        if self._price_func is None:
            raise RuntimeError('Power price not available')
        return self._price_func(times)

    def evaluate_at_time(self, time: dt.datetime) -> float:
        if self._price_func is None:
            raise RuntimeError('Power price not available')
        return float(self._price_func(time.timestamp()))

    def compute_total_price_between(self, start_time: float,
                                    end_time: float) -> float:
        if self._price_func is None:
            raise RuntimeError('Power price not available')
        assert end_time > start_time
        return self._price_func.integral(start_time, end_time)

    def compute_average_price_between(self, start_time: float,
                                      end_time: float) -> float:
        return self.compute_total_price_between(
            start_time, end_time) / (end_time - start_time)


class SimulatedPowerPrice(PowerPrice):

    def __init__(self,
                 duration_hours: float = 48.0,
                 interval_hours: float = 1.0,
                 period_hours: float = 24.0,
                 peak_hour: float = 17.0) -> None:
        super().__init__()
        self.duration_hours = duration_hours
        self.interval_hours = interval_hours
        self.period_hours = period_hours
        self.peak_hour = peak_hour
        self._unit = 'NOK/kWh'

    def fetch(self) -> None:
        self._fetch_time = dt.datetime.now()

        day_start_time = dt.datetime.combine(self._fetch_time.date(),
                                             dt.datetime.min.time())
        self._earliest_time = day_start_time
        self._latest_time = self._earliest_time + dt.timedelta(
            hours=self.duration_hours)

        price_peak_time = day_start_time + dt.timedelta(hours=self.peak_hour)

        period = dt.timedelta(hours=self.period_hours)
        n_samples = int(period / dt.timedelta(hours=self.interval_hours)) + 1

        self._times = np.linspace(self._earliest_time.timestamp(),
                                  self._latest_time.timestamp(), n_samples)

        self._prices = np.cos(2 * np.pi *
                              (self._times - price_peak_time.timestamp()) /
                              period.total_seconds())

        self._price_func = scipy.interpolate.UnivariateSpline(self._times,
                                                              self._prices,
                                                              s=0,
                                                              ext='const')


class TibberPowerPrice(PowerPrice):

    def __init__(self, access_token: str) -> None:
        super().__init__()
        self.access_token = access_token
        self._unit = 'NOK/kWh'

    def fetch(self) -> None:
        price_info = asyncio.run(self._fetch_price_info_or_none())
        if price_info is None or len(price_info) == 0:
            return

        timestrings, prices = zip(*price_info.items())
        datetimes = [
            dt.datetime.strptime(s, r'%Y-%m-%dT%H:%M:%S.%f%z')
            for s in timestrings
        ]

        self._earliest_time = datetimes[0]
        self._latest_time = datetimes[-1]

        self._times = datetimes_to_timestamps(datetimes)
        self._prices = np.asfarray(prices)

        self._price_func = scipy.interpolate.UnivariateSpline(self._times,
                                                              self._prices,
                                                              s=0,
                                                              ext='const')

    async def _fetch_price_info_or_none(self) -> Union[dict, None]:
        try:
            price_info = await self._fetch_price_info()
        except:
            price_info = None
        return price_info

    async def _fetch_price_info(self) -> dict:
        connection = tibber.Tibber(self.access_token)
        await connection.update_info()

        homes = connection.get_homes()
        if len(homes) == 0:
            raise RuntimeError('No homes for Tibber account')

        home = homes[0]
        await home.update_price_info()

        await connection.close_connection()

        return home.price_total


class TemperatureRecorder:

    def __init__(self,
                 smoothing_window_duration: float = 3000.0,
                 delay_min_heating_duration: float = 300.0,
                 delay_power_threshold: float = 0.5,
                 save_path: Optional[Union[str, Path]] = None) -> None:
        self.smoothing_window_duration = smoothing_window_duration
        self.delay_min_heating_duration = delay_min_heating_duration
        self.delay_power_threshold = delay_power_threshold
        self.save_path = None if save_path is None else Path(save_path)

        self._measurements: dict[str, list] = dict(times=[],
                                                   heating_powers=[],
                                                   temperatures=[],
                                                   outside_temperatures=[])
        self._data_arrays: Optional[dict[str, np.ndarray]] = None

    @property
    def n_measurements(self) -> int:
        return len(self._measurements['times'])

    @property
    def is_empty(self) -> bool:
        return self.n_measurements == 0

    @property
    def measurements(self) -> dict:
        return self._measurements

    @property
    def data_arrays(self) -> dict:
        if self._data_arrays is None:
            self._data_arrays = {
                name: np.asarray(measurements)
                for name, measurements in self.measurements.items()
            }
        return self._data_arrays

    def __enter__(self) -> None:
        self._load_measurements()

    def __exit__(self, *args: Any) -> None:
        self._save_measurements()

    def add_measurement(self, time: dt.datetime, heating_power: float,
                        temperature: float,
                        outside_temperature: float) -> None:
        timestamp = datetime_to_timestamp(time)
        assert len(self.measurements['times']
                   ) == 0 or timestamp > self.measurements['times'][-1]
        self.measurements['times'].append(timestamp)
        self.measurements['heating_powers'].append(heating_power)
        self.measurements['temperatures'].append(temperature)
        self.measurements['outside_temperatures'].append(outside_temperature)

        self._invalidate_data_arrays()

    def add_measurements(
            self, times: list, heating_powers: Union[list, np.ndarray],
            temperatures: Union[list, np.ndarray],
            outside_temperatures: Union[list, np.ndarray]) -> None:
        assert len(heating_powers) == len(times)
        assert len(temperatures) == len(times)
        assert len(outside_temperatures) == len(times)

        self.measurements['times'].extend(datetimes_to_timestamps(times))
        self.measurements['heating_powers'].extend(list(heating_powers))
        self.measurements['temperatures'].extend(list(temperatures))
        self.measurements['outside_temperatures'].extend(
            list(outside_temperatures))

        self._invalidate_data_arrays()

    def remove_measurements_prior_to_time(self,
                                          limit_time: dt.datetime) -> None:
        limit_timestamp = limit_time.timestamp()
        self._measurements = dict(
            zip(
                self.measurements.keys(),
                map(
                    list,
                    zip(*(properties
                          for properties in zip(*self.measurements.values())
                          if properties[0] < limit_timestamp)))))

        self._invalidate_data_arrays()

    def create_processed_record(self) -> 'ProcessedTemperatureRecord':
        return ProcessedTemperatureRecord(
            self.data_arrays['times'],
            self.data_arrays['heating_powers'],
            self.data_arrays['temperatures'],
            self.data_arrays['outside_temperatures'],
            smoothing_window_duration=self.smoothing_window_duration,
            delay_min_heating_duration=self.delay_min_heating_duration,
            delay_power_threshold=self.delay_power_threshold)

    def plot_evolution(self,
                       convert_timestamps: Callable = timestamps_to_datetimes,
                       output_path: Optional[Union[str, Path]] = None) -> None:
        times = convert_timestamps(self.data_arrays['times'])
        heating_powers = self.data_arrays['heating_powers']
        temperatures = self.data_arrays['temperatures']
        outside_temperatures = self.data_arrays['outside_temperatures']

        plot(dict(x_values=times,
                  y_values=temperatures,
                  marker='.',
                  label='Temperature inside'),
             dict(x_values=times,
                  y_values=outside_temperatures,
                  marker='.',
                  label='Temperature outside'),
             dict(ax=1,
                  x_values=times,
                  y_values=heating_powers,
                  marker='.',
                  label='Heating'),
             x_uses_datetimes=True,
             ylabel='Temperature [°C]',
             second_ylabel='Heating power [W]',
             output_path=output_path)

    def _invalidate_data_arrays(self) -> None:
        self._data_arrays = None

    def _load_measurements(self) -> None:
        if self.save_path is not None and self.save_path.exists():
            self._data_arrays = np.load(self.save_path)
            for name in self._measurements:
                self._measurements[name] = list(self._data_arrays[name])

    def _save_measurements(self) -> None:
        if self.save_path is not None and not self.is_empty:
            np.savez_compressed(self.save_path, **self.data_arrays)


class ProcessedTemperatureRecord:

    def __init__(
        self,
        times: np.ndarray,
        heating_powers: np.ndarray,
        temperatures: np.ndarray,
        outside_temperatures: np.ndarray,
        smoothing_window_duration: float = 3000.0,
        delay_min_heating_duration: float = 300.0,
        delay_power_threshold: float = 0.5,
    ) -> None:
        assert times.ndim == 1
        assert heating_powers.shape == times.shape
        assert temperatures.shape == times.shape
        assert outside_temperatures.shape == times.shape

        self.times = times
        self.heating_powers = heating_powers
        self.temperatures = temperatures
        self.outside_temperatures = outside_temperatures

        self.smoothing_window_duration = smoothing_window_duration
        self.delay_min_heating_duration = delay_min_heating_duration
        self.delay_power_threshold = delay_power_threshold

        self.temperature_rates = self.__class__.compute_smooth_rate_of_change(
            self.times,
            self.temperatures,
            smoothing_window_duration=self.smoothing_window_duration)

        self.corrected_heating_powers = self.heating_powers

        self.temperature_diffs = self.temperatures - self.outside_temperatures

    @property
    def earliest_timestamp(self) -> float:
        return self.times[0]

    @property
    def latest_timestamp(self) -> float:
        return self.times[-1]

    def detect_heating_delay(self, fallback_value: float = 0.0) -> float:
        heating_delay = self.__class__._estimate_heating_delay(
            self.times, self.heating_powers, self.temperature_rates,
            self.delay_min_heating_duration, self.delay_power_threshold)

        if heating_delay is None:
            heating_delay = fallback_value

        self.corrected_heating_powers = self.__class__._shift_heating_by_delay(
            self.times, self.heating_powers, heating_delay)

        return heating_delay

    def find_heating_periods(
            self) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], None]:
        result = self.__class__._find_heating_period_indices(
            self.times, self.heating_powers, self.delay_power_threshold)

        if result is None:
            return None
        else:
            heating_started_indices, heating_ended_indices = result
            start_times = self.times[heating_started_indices]
            end_times = self.times[heating_ended_indices]
            mean_heating_powers = np.asfarray([
                np.mean(self.heating_powers[start:end + 1]) for start, end in
                zip(heating_started_indices, heating_ended_indices)
            ])
            return mean_heating_powers, start_times, end_times

    def add_to_table(self, table: 'MeasurementTable') -> None:
        table.add_measurements(self.corrected_heating_powers,
                               self.temperature_diffs, self.temperature_rates)

    def plot_evolution(self,
                       *args,
                       convert_timestamps: Callable = timestamps_to_datetimes,
                       heating_power_threshold: float = 50.0,
                       **kwargs) -> None:
        times = convert_timestamps(self.times)
        marker = None

        plot(dict(x_values=times,
                  y_values=self.temperatures,
                  label='Temperature inside'),
             dict(x_values=times,
                  y_values=self.outside_temperatures,
                  label='Temperature outside'),
             dict(ax=1,
                  x_values=times,
                  y_values=self.temperature_rates *
                  dt.timedelta(hours=1).total_seconds(),
                  label='Change inside'),
             dict(type='fill',
                  x_values=times,
                  y_values=self.corrected_heating_powers,
                  y_threshold=heating_power_threshold,
                  color='tab:red',
                  alpha=0.2),
             *args,
             x_uses_datetimes=True,
             xlim=(times[0], times[-1]),
             ylabel='Temperature [°C]',
             second_ylabel='Rate of change [°C/hour]',
             **kwargs)

    def plot_scatter(self, **kwargs) -> None:
        plot(dict(type='scatter',
                  x_values=self.temperature_diffs,
                  y_values=self.temperature_rates *
                  dt.timedelta(hours=1).total_seconds(),
                  c=self.corrected_heating_powers,
                  s=10,
                  colorbar=True,
                  clabel='Heating power [W]'),
             xlabel='Temperature difference [°C]',
             ylabel='Temperature rate of change [°C/hour]',
             **kwargs)

    @classmethod
    def compute_smooth_evolution(
            cls, times: np.ndarray, values: np.ndarray,
            smoothing_window_duration: float) -> np.ndarray:
        assert times.shape == values.shape
        assert smoothing_window_duration >= 0

        regular_times = np.linspace(times[0], times[-1], times.size)
        smoothing_window_length = max(
            1,
            int(
                np.ceil(smoothing_window_duration /
                        (regular_times[1] - regular_times[0]))))

        values_at_regular_times = cls._resample(times, values, regular_times)
        smoothed_values_at_regular_times = cls._smooth(
            values_at_regular_times, smoothing_window_length)
        values = cls._resample(regular_times, smoothed_values_at_regular_times,
                               times)
        return values

    @classmethod
    def compute_smooth_rate_of_change(
            cls, times: np.ndarray, values: np.ndarray,
            smoothing_window_duration: float) -> np.ndarray:
        assert times.shape == values.shape
        assert smoothing_window_duration >= 0

        regular_times = np.linspace(times[0], times[-1], times.size)
        smoothing_window_length = max(
            1,
            int(
                np.ceil(smoothing_window_duration /
                        (regular_times[1] - regular_times[0]))))

        values_at_regular_times = cls._resample(times, values, regular_times)
        smoothed_values_at_regular_times = cls._smooth(
            values_at_regular_times, smoothing_window_length)
        gradients_at_regular_times = np.gradient(
            smoothed_values_at_regular_times, regular_times)
        gradients = cls._resample(regular_times, gradients_at_regular_times,
                                  times)
        return gradients

    @staticmethod
    def _resample(times: np.ndarray, values: np.ndarray,
                  new_times: np.ndarray) -> np.ndarray:
        return scipy.interpolate.interp1d(times,
                                          values,
                                          kind='linear',
                                          copy=False,
                                          assume_sorted=True)(new_times)

    @staticmethod
    def _smooth(values: np.ndarray,
                window_length: int,
                order: int = 3) -> np.ndarray:
        window_length = (window_length +
                         1) if window_length % 2 == 0 else window_length
        max_length = (values.size - 1) if values.size % 2 == 0 else values.size
        min_length = (order + 1) if order % 2 == 0 else order + 2
        window_length = min(max_length, window_length)
        if window_length < min_length:
            return values
        else:
            return scipy.signal.savgol_filter(values, window_length, order)

    @staticmethod
    def _find_heating_period_indices(
        times: np.ndarray,
        heating_powers: np.ndarray,
        power_threshold: float,
        include_initial_heating: bool = True
    ) -> Union[Tuple[np.ndarray, np.ndarray], None]:
        assert times.ndim == 1
        assert heating_powers.shape == times.shape
        assert power_threshold >= 0

        is_heating = heating_powers > power_threshold * heating_powers.max()

        if np.all(np.logical_not(is_heating)):
            return None

        if np.all(is_heating):
            if include_initial_heating:
                return np.array([0]), np.array([times.size - 1])
            else:
                return None

        heating_started_indices = np.nonzero(
            is_heating[1:] > is_heating[:-1])[0] + 1
        heating_ended_indices = np.nonzero(is_heating[1:] < is_heating[:-1])[0]

        if is_heating[0] and not include_initial_heating:
            heating_ended_indices = heating_ended_indices[1:]
        if is_heating[-1]:
            heating_ended_indices = np.append(heating_ended_indices,
                                              times.size - 1)

        return heating_started_indices, heating_ended_indices

    @staticmethod
    def _estimate_heating_delay(times: np.ndarray, heating_powers: np.ndarray,
                                temperature_rates: np.ndarray,
                                min_heating_duration: float,
                                power_threshold: float) -> Union[float, None]:
        assert times.ndim == 1
        assert heating_powers.shape == times.shape
        assert temperature_rates.shape == times.shape
        assert min_heating_duration >= 0
        assert power_threshold >= 0

        heating_delay = None

        result = ProcessedTemperatureRecord._find_heating_period_indices(
            times,
            heating_powers,
            power_threshold,
            include_initial_heating=False)
        if result is None:
            return heating_delay

        heating_started_indices, heating_ended_indices = result

        heating_delays = []
        for start_idx, end_idx in zip(heating_started_indices,
                                      heating_ended_indices):
            start_time = times[start_idx]
            end_time = times[end_idx]
            duration = end_time - start_time

            if duration < min_heating_duration:
                continue

            peak_idx = start_idx + np.argmax(
                np.gradient(temperature_rates)[start_idx:end_idx + 1])

            heating_delays.append(times[peak_idx] - start_time)

        if len(heating_delays) > 0:
            heating_delay = np.mean(heating_delays)

        return heating_delay

    @staticmethod
    def _shift_heating_by_delay(times: np.ndarray, heating_powers: np.ndarray,
                                heating_delay: float) -> np.ndarray:
        assert heating_delay >= 0

        if heating_delay == 0:
            return heating_powers
        else:
            heating_power_func = scipy.interpolate.interp1d(
                times,
                heating_powers,
                kind='nearest',
                fill_value='extrapolate',
                copy=False,
                assume_sorted=True)
            return heating_power_func(times - heating_delay)


class MeasurementTable:

    @staticmethod
    def new(max_heating_power: float = 2500.0,
            n_heating_powers: int = 26,
            min_temperature_diff: float = 0.0,
            max_temperature_diff: float = 50.0,
            n_temperature_diffs: int = 51,
            n_measurements_per_cell: int = 10,
            **kwargs) -> 'MeasurementTable':
        temperature_rate_table = np.full(
            (n_heating_powers, n_temperature_diffs, n_measurements_per_cell),
            np.nan)
        depth_index_table = np.zeros((n_heating_powers, n_temperature_diffs),
                                     dtype=int)

        return MeasurementTable(max_heating_power, min_temperature_diff,
                                max_temperature_diff, temperature_rate_table,
                                depth_index_table, **kwargs)

    @staticmethod
    def from_file(save_path: Union[str, Path]) -> 'MeasurementTable':
        data = np.load(save_path)
        return MeasurementTable(data['max_heating_power'].item(),
                                data['min_temperature_diff'].item(),
                                data['max_temperature_diff'].item(),
                                data['temperature_rate_table'],
                                data['depth_index_table'],
                                save_path=save_path)

    def __init__(self,
                 max_heating_power: float,
                 min_temperature_diff: float,
                 max_temperature_diff: float,
                 temperature_rate_table: np.ndarray,
                 depth_index_table: np.ndarray,
                 save_path: Optional[Union[str, Path]] = None) -> None:
        assert temperature_rate_table.ndim == 3
        assert depth_index_table.shape == temperature_rate_table.shape[:-1]

        self.max_heating_power = max_heating_power
        self.min_temperature_diff = min_temperature_diff
        self.max_temperature_diff = max_temperature_diff
        self.temperature_rate_table = temperature_rate_table
        self.depth_index_table = depth_index_table
        self.save_path = None if save_path is None else Path(save_path)

        self.heating_power_interval = self.max_heating_power / temperature_rate_table.shape[
            0]
        self.temperature_diff_interval = (
            self.max_temperature_diff -
            self.min_temperature_diff) / temperature_rate_table.shape[1]

        self.heating_power_bin_edges = np.linspace(
            0, self.max_heating_power, self.temperature_rate_table.shape[0])
        self.temperature_diff_bin_edges = np.linspace(
            self.min_temperature_diff, self.max_temperature_diff,
            self.temperature_rate_table.shape[1])

    @property
    def fitting_possible(self) -> bool:
        return self.__class__._fitting_possible_with_valid_mask(
            np.any(np.isfinite(self.temperature_rate_table), axis=-1))

    def __enter__(self) -> 'MeasurementTable':
        return self

    def __exit__(self, *args: Any) -> None:
        self.save()

    def save(self) -> None:
        if self.save_path is not None:
            np.savez_compressed(
                self.save_path,
                max_heating_power=self.max_heating_power,
                min_temperature_diff=self.min_temperature_diff,
                max_temperature_diff=self.max_temperature_diff,
                temperature_rate_table=self.temperature_rate_table,
                depth_index_table=self.depth_index_table)

    def add_measurement(self, heating_power: float, temperature_diff: float,
                        temperature_rate: float) -> None:
        heating_power_idx = max(
            0,
            min(self.temperature_rate_table.shape[0] - 1,
                int(heating_power // self.heating_power_interval)))
        temperature_diff_idx = max(
            0,
            min(
                self.temperature_rate_table.shape[1] - 1,
                int((temperature_diff - self.min_temperature_diff) //
                    self.temperature_diff_interval)))

        depth_idx = self.depth_index_table[heating_power_idx,
                                           temperature_diff_idx]

        self.temperature_rate_table[heating_power_idx, temperature_diff_idx,
                                    depth_idx] = temperature_rate
        self.depth_index_table[heating_power_idx, temperature_diff_idx] = (
            depth_idx + 1) % self.temperature_rate_table.shape[2]

    def add_measurements(self, heating_powers: np.ndarray,
                         temperature_diffs: np.ndarray,
                         temperature_rates: np.ndarray) -> None:
        assert heating_powers.shape == temperature_rates.shape
        assert temperature_diffs.shape == temperature_rates.shape
        assert temperature_rates.ndim == 1

        heating_power_indices = np.maximum(
            0,
            np.minimum(self.temperature_rate_table.shape[0] - 1,
                       (heating_powers //
                        self.heating_power_interval).astype(int)))
        temperature_diff_indices = np.maximum(
            0,
            np.minimum(self.temperature_rate_table.shape[1] - 1,
                       ((temperature_diffs - self.min_temperature_diff) //
                        self.temperature_diff_interval).astype(int)))

        flat_depth_index_table = np.ravel(self.depth_index_table)
        flat_temperature_rate_table = self.temperature_rate_table.reshape(
            -1, self.temperature_rate_table.shape[-1])

        indices_to_flat_tables = np.ravel_multi_index(
            (heating_power_indices, temperature_diff_indices),
            self.depth_index_table.shape)

        for i in range(temperature_rates.size):
            idx_to_flat_tables = indices_to_flat_tables[i]

            depth_idx = flat_depth_index_table[idx_to_flat_tables]

            flat_temperature_rate_table[idx_to_flat_tables,
                                        depth_idx] = temperature_rates[i]

            flat_depth_index_table[idx_to_flat_tables] = (
                depth_idx + 1) % self.temperature_rate_table.shape[2]

    def create_model_from_fit(self,
                              heating_delay: float = 0.0,
                              **kwargs) -> 'NewtonHeatingModel':
        return NewtonHeatingModel(heating_delay, *self._fit_measurements(),
                                  **kwargs)

    def plot(self, include_fit: bool = True, **kwargs) -> None:
        if include_fit and self.fitting_possible:
            model = self.create_model_from_fit()
        else:
            model = None

        temperature_rates = np.nanmean(self.temperature_rate_table, axis=-1)
        weights = np.sum(np.isfinite(self.temperature_rate_table),
                         axis=-1) / self.temperature_rate_table.shape[-1]
        valid_mask = np.isfinite(temperature_rates)

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_idx = 0

        artists = []

        for idx in range(temperature_rates.shape[0]):
            if not np.any(valid_mask[idx, :]):
                continue

            valid_temperature_diffs = self.temperature_diff_bin_edges[
                valid_mask[idx, :]]
            valid_temperature_rates = temperature_rates[idx, :][valid_mask[
                idx, :]]
            valid_weights = weights[idx, :][valid_mask[idx, :]]

            artists.append(
                dict(type='scatter',
                     x_values=valid_temperature_diffs,
                     y_values=valid_temperature_rates *
                     dt.timedelta(hours=1).total_seconds(),
                     s=20 * valid_weights,
                     c=colors[color_idx],
                     label=f'{self.heating_power_bin_edges[idx]:g} W'))

            if model is not None and np.sum(valid_mask[idx, :]) > 3:
                artists.append(
                    dict(x_values=valid_temperature_diffs,
                         y_values=model(self.heating_power_bin_edges[idx],
                                        valid_temperature_diffs) *
                         dt.timedelta(hours=1).total_seconds(),
                         ls='--',
                         color=colors[color_idx]))

            color_idx += 1

        plot(*artists,
             xlabel='Temperature difference [°C]',
             ylabel='Temperature rate of change [°C/hour]',
             **kwargs)

    @staticmethod
    def _fitting_possible_with_valid_mask(valid_mask: np.ndarray) -> bool:
        return np.sum(np.any(valid_mask, axis=0)) > 1 and np.sum(
            np.any(valid_mask, axis=1)) > 1

    def _fit_measurements(self) -> Tuple[float, float, float]:
        weights = np.sum(np.isfinite(self.temperature_rate_table),
                         axis=-1) / self.temperature_rate_table.shape[-1]
        temperature_rates = np.nanmean(self.temperature_rate_table, axis=-1)

        valid_mask = np.isfinite(temperature_rates)

        if not self.__class__._fitting_possible_with_valid_mask(valid_mask):
            raise RuntimeError('Insufficient data for fitting')

        heating_powers, temperature_diffs = np.meshgrid(
            self.heating_power_bin_edges,
            self.temperature_diff_bin_edges,
            indexing='ij')

        valid_weights = weights[valid_mask].ravel()
        valid_temperature_rates = temperature_rates[valid_mask].ravel()
        valid_coordinates = np.stack((heating_powers[valid_mask].ravel(),
                                      temperature_diffs[valid_mask].ravel()),
                                     axis=1)

        regression = sklearn.linear_model.LinearRegression()
        regression.fit(valid_coordinates,
                       valid_temperature_rates,
                       sample_weight=valid_weights)

        return regression.coef_[0], regression.intercept_, regression.coef_[1]


class NewtonHeatingModel:

    @staticmethod
    def from_file(save_path: Union[str, Path]) -> 'NewtonHeatingModel':
        data = np.load(save_path)
        return NewtonHeatingModel(data['heating_delay'].item(),
                                  data['heating_source_coef'].item(),
                                  data['source_term'].item(),
                                  data['transfer_coef'].item())

    def __init__(self,
                 heating_delay: float,
                 heating_source_coef: float,
                 source_term: float,
                 transfer_coef: float,
                 save_path: Optional[Union[str, Path]] = None) -> None:
        self.heating_delay = heating_delay
        self.heating_source_coef = heating_source_coef
        self.source_term = source_term
        self.transfer_coef = transfer_coef
        self.save_path = None if save_path is None else Path(save_path)

    @property
    def heating_delay(self) -> float:
        return self._heating_delay

    @property
    def heating_source_coef(self) -> float:
        return self._heating_source_coef

    @property
    def source_term(self) -> float:
        return self._source_term

    @property
    def transfer_coef(self) -> float:
        return self._transfer_coef

    @heating_delay.setter
    def heating_delay(self, value: float) -> None:
        assert value >= 0
        self._heating_delay = value

    @heating_source_coef.setter
    def heating_source_coef(self, value: float) -> None:
        assert value > 0
        self._heating_source_coef = value

    @source_term.setter
    def source_term(self, value: float) -> None:
        assert value > 0
        self._source_term = value

    @transfer_coef.setter
    def transfer_coef(self, value: float) -> None:
        assert value < 0
        self._transfer_coef = value

    def __enter__(self) -> 'NewtonHeatingModel':
        return self

    def __exit__(self, *args: Any) -> None:
        self.save()

    def save(self) -> None:
        if self.save_path is not None:
            np.savez(self.save_path,
                     heating_delay=self.heating_delay,
                     heating_source_coef=self.heating_source_coef,
                     source_term=self.source_term,
                     transfer_coef=self.transfer_coef)

    def __call__(
        self, heating_power: float,
        temperature_diff: Union[float,
                                np.ndarray]) -> Union[float, np.ndarray]:
        return self.source_term + self.heating_source_coef * heating_power + self.transfer_coef * temperature_diff

    def _compute_evolution_numerically(
            self,
            times: np.ndarray,
            outside_temperatures: np.ndarray,
            initial_temperature: float,
            heating_power: float,
            heating_start_time: Optional[float] = None,
            heating_duration: Optional[float] = None) -> np.ndarray:
        assert times.shape == outside_temperatures.shape

        initial_time = times[0]
        final_time = times[-1]

        heating_start = initial_time if heating_start_time is None else heating_start_time
        heating_end = final_time if heating_duration is None else (
            heating_start + heating_duration)

        heating_start = max(
            initial_time, min(final_time, heating_start + self.heating_delay))
        heating_end = max(heating_start,
                          min(final_time, heating_end + self.heating_delay))

        temperatures = np.zeros_like(outside_temperatures)
        temperatures[0] = initial_temperature

        for i in range(times.size - 1):
            time = times[i]
            source_term = self.source_term if (
                time < heating_start or time > heating_end) else (
                    self.heating_source_coef * heating_power +
                    self.source_term)
            temperatures[i + 1] = temperatures[i] + (
                source_term + self.transfer_coef *
                (temperatures[i] - outside_temperatures[i])) * (times[i + 1] -
                                                                time)

        return temperatures

    def compute_evolution_with_thermostat(
            self,
            times: np.ndarray,
            outside_temperatures: np.ndarray,
            initial_temperature: float,
            thermostat_mode_a_min_temperature: float,
            thermostat_mode_b_min_temperature: float,
            thermostat_mode_a_start_indices: np.ndarray,
            thermostat_mode_a_end_indices: np.ndarray,
            logger=print,
            **kwargs):
        assert thermostat_mode_a_start_indices.ndim == 1
        assert thermostat_mode_a_start_indices.shape == thermostat_mode_a_end_indices.shape

        if thermostat_mode_a_start_indices.size == 0:
            starts_in_mode_b = True
            transition_indices = np.array([0, times.size - 1], dtype=int)
        else:
            starts_in_mode_b = thermostat_mode_a_start_indices[
                0] > 0 and thermostat_mode_a_start_indices[
                    0] <= thermostat_mode_a_end_indices[0]

            n_start_indices = thermostat_mode_a_start_indices.size
            n_end_indices = thermostat_mode_a_end_indices.size
            has_exterior_start_idx = int(
                thermostat_mode_a_start_indices[0] == 0)
            has_exterior_end_idx = int(
                thermostat_mode_a_end_indices[-1] == times.size - 1)

            transition_indices = np.zeros(n_start_indices + n_end_indices +
                                          (1 - has_exterior_start_idx) +
                                          (1 - has_exterior_end_idx),
                                          dtype=int)
            transition_indices[0] = 0
            transition_indices[-1] = times.size - 1

            if starts_in_mode_b:
                transition_indices[1:-2:2] = thermostat_mode_a_start_indices
                transition_indices[2:-1:2] = thermostat_mode_a_end_indices
            else:
                transition_indices[1:-2:2] = thermostat_mode_a_end_indices[:n_end_indices -
                                                       has_exterior_end_idx]
                transition_indices[
                    2:-1:
                    2] = thermostat_mode_a_start_indices[
                    has_exterior_start_idx:]

        thermostat_min_temperatures = np.zeros(transition_indices.size - 1,
                                               dtype=float)
        if starts_in_mode_b:
            thermostat_min_temperatures[
                :-1:2] = thermostat_mode_b_min_temperature
            thermostat_min_temperatures[
                1::2] = thermostat_mode_a_min_temperature
        else:
            thermostat_min_temperatures[
                :-1:2] = thermostat_mode_a_min_temperature
            thermostat_min_temperatures[
                1::2] = thermostat_mode_b_min_temperature

        slice_initial_temperature = initial_temperature

        temperatures = np.zeros_like(times)

        for thermostat_min_temperature, start_idx, end_idx in zip(
                thermostat_min_temperatures, transition_indices[:-1],
                transition_indices[1:]):
            thermostat_time_slice = slice(start_idx, end_idx + 1)
            temperatures[thermostat_time_slice] = np.maximum(
                thermostat_min_temperature,
                self.compute_evolution(
                    times[thermostat_time_slice],
                    outside_temperatures[thermostat_time_slice],
                    slice_initial_temperature, **kwargs))
            slice_initial_temperature = temperatures[end_idx]

        return temperatures

    def compute_evolution(self,
                          times: np.ndarray,
                          outside_temperatures: np.ndarray,
                          initial_temperature: float,
                          heating_powers: Optional[np.ndarray] = None,
                          heating_start_times: Optional[np.ndarray] = None,
                          heating_end_times: Optional[np.ndarray] = None) -> np.ndarray:
        assert times.shape == outside_temperatures.shape

        initial_time = times[0]
        final_time = times[-1]

        times = times - initial_time
        exp_factor = np.exp(self.transfer_coef * times)
        inv_exp_factor = 1 / exp_factor

        temperatures = (initial_temperature * exp_factor) - (
            self.transfer_coef * exp_factor *
            scipy.integrate.cumulative_trapezoid(
                outside_temperatures * inv_exp_factor, x=times, initial=0)
        ) + exp_factor * (self.source_term *
                          scipy.integrate.cumulative_trapezoid(
                              inv_exp_factor, x=times, initial=0))

        if heating_powers is not None or heating_start_times is not None or heating_end_times is not None:
            assert heating_powers is not None and heating_start_times is not None and heating_end_times is not None
            assert heating_start_times.shape == heating_powers.shape
            assert heating_end_times.shape == heating_powers.shape

            for heating_power, heating_start_time, heating_end_time in zip(
                    heating_powers, heating_start_times, heating_end_times):

                heating_start_time = max(
                    initial_time,
                    min(final_time, heating_start_time + self.heating_delay))
                heating_end_time = max(
                    heating_start_time,
                    min(final_time, heating_end_time + self.heating_delay))

                if heating_start_time == heating_end_time:
                    continue

                heating_start_time -= initial_time
                heating_end_time -= initial_time

                heating_time_start_idx = np.searchsorted(times,
                                                         heating_start_time,
                                                         side='left')
                heating_time_end_idx = np.searchsorted(times,
                                                       heating_end_time,
                                                       side='right')

                if heating_time_end_idx > heating_time_start_idx:
                    heating_time_slice = slice(heating_time_start_idx,
                                               heating_time_end_idx)

                    heating_contribution = (
                        self.heating_source_coef *
                        heating_power) * scipy.integrate.cumulative_trapezoid(
                            inv_exp_factor[heating_time_slice],
                            x=times[heating_time_slice],
                            initial=0)

                    temperatures[heating_time_slice] += exp_factor[
                        heating_time_slice] * heating_contribution
                    temperatures[heating_time_end_idx:] += exp_factor[
                        heating_time_end_idx:] * heating_contribution[-1]

        return temperatures

    def _compute_equilibrium_temperature(self, outside_temperature: float,
                                         heating_power: float) -> float:
        return outside_temperature - (self.heating_source_coef * heating_power
                                      + self.source_term) / self.transfer_coef

    def _compute_heater_state_integral_value(self, times: np.ndarray,
                                             outside_temperatures: np.ndarray,
                                             initial_temperature: float,
                                             final_temperature: float,
                                             heating_power: float) -> float:
        assert times.shape == outside_temperatures.shape

        times = times - times[0]
        inv_exp_factor = np.exp(-self.transfer_coef * times)
        final_inv_exp_factor = inv_exp_factor[-1]

        return (final_temperature * final_inv_exp_factor -
                initial_temperature + self.transfer_coef *
                np.trapz(outside_temperatures * inv_exp_factor, x=times) +
                self.source_term *
                (final_inv_exp_factor - 1.0) / self.transfer_coef) / (
                    self.heating_source_coef * heating_power)

    def _compute_required_heating_duration_from_start_time(
            self,
            times: np.ndarray,
            outside_temperatures: np.ndarray,
            initial_temperature: float,
            final_temperature: float,
            heating_power: float,
            heating_start_time: float,
            integral_value: Optional[float] = None) -> Optional[float]:
        assert heating_start_time >= times[0]
        assert heating_start_time <= times[-1]

        integral_value = self._compute_heater_state_integral_value(
            times, outside_temperatures, initial_temperature,
            final_temperature,
            heating_power) if integral_value is None else integral_value

        if integral_value <= 0:
            return 0.0

        heating_start_time += self.heating_delay

        heating_duration = -np.log(1.0 - self.transfer_coef *
                                   np.exp(self.transfer_coef *
                                          (heating_start_time - times[0])) *
                                   integral_value) / self.transfer_coef

        if heating_start_time + heating_duration > times[-1]:
            return None
        else:
            return heating_duration

    def _compute_required_heating_duration_until_end_time(
            self,
            times: np.ndarray,
            outside_temperatures: np.ndarray,
            initial_temperature: float,
            final_temperature: float,
            heating_power: float,
            heating_end_time: float,
            integral_value: Optional[float] = None) -> Optional[float]:

        assert heating_end_time >= times[0]
        assert heating_end_time <= times[-1]

        integral_value = self._compute_heater_state_integral_value(
            times, outside_temperatures, initial_temperature,
            final_temperature,
            heating_power) if integral_value is None else integral_value

        if integral_value <= 0:
            return 0.0

        heating_end_time += self.heating_delay

        heating_duration = np.log(1.0 + self.transfer_coef *
                                  np.exp(self.transfer_coef *
                                         (heating_end_time - times[0])) *
                                  integral_value) / self.transfer_coef

        if heating_end_time - heating_duration < times[0]:
            return None
        else:
            return heating_duration


if __name__ == '__main__':
    power_price = PowerPrice()
    print(power_price(dt.datetime.now()))

    heating_power = 0.2
    heating_model = NewtonHeatingModel(1.5, 1.2, 0.1, -0.5)
    times = np.linspace(2.0, 12.0, 500)
    outside_temperatures = np.full_like(times, 0.0)
    initial_temperature = 1.0
    heating_start_time = 2.487
    heating_duration = 2.0
    temperatures_numerical = heating_model._compute_evolution_numerically(
        times, outside_temperatures, initial_temperature, heating_power,
        heating_start_time, heating_duration)
    temperatures_analytical = heating_model.compute_evolution(
        times, outside_temperatures, initial_temperature, heating_power,
        heating_start_time, heating_duration)
    print(temperatures_analytical[-1])

    required_heating_duration = heating_model._compute_required_heating_duration(
        times, outside_temperatures, initial_temperature, 0.1, heating_power,
        heating_start_time)
    print(required_heating_duration)

    fig, ax = plt.subplots(dpi=200)
    ax.plot(times, temperatures_numerical, label='Numerical')
    ax.plot(times, temperatures_analytical, ls='--', label='Analytical')
    ax.legend(loc='best')
    plt.show()