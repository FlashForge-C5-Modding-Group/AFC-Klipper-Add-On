import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from extras.AFC import afc
from extras.AFC_lane import AFCLane
from tests.conftest import MockAFC, MockConfig, MockLogger, MockPrinter


@pytest.fixture
def make_afc(monkeypatch):
    monkeypatch.setattr('extras.AFC.AFC_logger', lambda printer, owner: MockLogger())
    monkeypatch.setattr(afc, 'print_version', lambda self: None)

    def make(values=None):
        printer = MockPrinter()
        config = MockConfig(name='AFC', printer=printer, values=values)
        # The file writer is unrelated to these tests and must not outlive them.
        with patch('extras.AFC.threading.Thread'):
            obj = afc(config)
        return obj, printer

    return make


class TestResetFileCallback:
    def test_disabled_metadata_resets_print_without_scheduling_query(self, make_afc):
        obj, printer = make_afc({'enable_print_metadata': False})
        reactor = printer.get_reactor()
        reactor.register_timer = MagicMock()
        obj.moonraker = MagicMock()
        obj.print_data_metadata = MagicMock()
        obj.number_of_toolchanges = 8
        obj.current_toolchange = 4
        obj.print_tool_temperatures = [220]
        obj.save_vars = MagicMock()

        obj._reset_file_callback()

        assert obj.in_print_timer is None
        reactor.register_timer.assert_not_called()
        obj.print_data_metadata.query_filename.assert_not_called()
        obj.moonraker.get_file_metadata.assert_not_called()
        assert obj.number_of_toolchanges == 0
        assert obj.current_toolchange == -1
        assert obj.print_tool_temperatures == []
        obj.print_data_metadata.reset.assert_called_once_with()


class TestTd1Present:
    def test_disabled_detection_does_not_schedule_or_query(self, make_afc):
        obj, printer = make_afc({'enable_td1_detection': False})
        obj._td1_present = True
        obj.moonraker = MagicMock()
        reactor = printer.get_reactor()
        reactor.register_async_callback = MagicMock()

        assert obj.td1_present is False
        assert obj.check_td1_present() is False
        assert obj._schedule_td1_refresh() is None
        reactor.register_async_callback.assert_not_called()
        obj.moonraker.check_for_td1.assert_not_called()
        assert obj._td1_present is True


class TestHandleMoonrakerConnect:
    def test_disabled_detection_skips_startup_query(self, make_afc, monkeypatch):
        obj, printer = make_afc({'enable_td1_detection': False})
        obj.toolhead = MagicMock()
        obj.moonraker = MagicMock()
        obj.moonraker.wait_for_moonraker.return_value = True
        obj.moonraker.get_spoolman_server.return_value = None
        monkeypatch.setattr('extras.AFC.AFCStats', MagicMock())
        monkeypatch.setattr('extras.AFC.AFC_PrintFileMetaData', MagicMock())
        connected = []
        printer.register_event_handler('afc:moonraker_connect', lambda: connected.append(True))

        obj.handle_moonraker_connect()

        assert connected == [True]
        assert obj.td1_defined is False
        assert obj._td1_present is False
        assert obj.lane_data_enabled is False
        obj.moonraker.check_for_td1.assert_not_called()


class TestWeightTimers:
    def test_disabled_tracking_never_schedules_or_updates_weight(self):
        owner = MockAFC()
        owner.track_filament_weight = False
        printer = MockPrinter(afc=owner)
        reactor = printer.get_reactor()
        reactor.register_timer = MagicMock()
        reactor.update_timer = MagicMock()
        config = MockConfig(name='AFC_stepper lane1', printer=printer,
                            values={'unit': 'Turtle_1:1'})
        with patch('extras.AFC_lane.add_filament_switch',
                   return_value=(MagicMock(), MagicMock())):
            lane = AFCLane(config)
        lane.past_extruder_position = 42
        lane.save_counter = 7
        lane.weight = 300

        lane.enable_weight_timer()
        lane.disable_weight_timer()

        assert lane.cb_update_weight is None
        # The lane's espooler registers its own two timers; no weight timer is registered
        # or scheduled.
        registered = [c.args[0] for c in reactor.register_timer.call_args_list]
        assert registered == [lane.espooler.timer_callback, lane.espooler.timer_stats_callback]
        reactor.update_timer.assert_not_called()
        owner.function.get_extruder_pos.assert_not_called()
        owner.save_vars.assert_not_called()
        assert lane.past_extruder_position == 42
        assert lane.save_counter == 7
        assert lane.weight == 300


class TestAfcInit:
    def test_auto_switch_requires_weight_tracking(self, make_afc):
        from configfile import error

        with pytest.raises(error, match='auto_spool_switch requires track_filament_weight'):
            make_afc({'auto_spool_switch': True, 'track_filament_weight': False})

    def test_varfile_expands_home(self, make_afc, monkeypatch, tmp_path):
        monkeypatch.setenv('HOME', str(tmp_path))
        monkeypatch.setenv('USERPROFILE', str(tmp_path))
        obj, _ = make_afc({'VarFile': '~/afc/state.var'})
        assert os.path.normpath(obj.VarFile) == os.path.normpath(str(tmp_path / 'afc/state.var'))


class TestUpdateTrsync:
    def test_kalico_package_singleton_is_updated_not_a_top_level_copy(self, make_afc,
                                                                       monkeypatch):
        # Kalico's loaded [danger_options] lives in klippy.extras.danger_options; a separately
        # executed top-level copy holds its own, unused options object.
        loaded = types.SimpleNamespace(multi_mcu_trsync_timeout=0.025,
                                       single_mcu_trsync_timeout=0.25)
        stale = types.SimpleNamespace(multi_mcu_trsync_timeout=0.025,
                                      single_mcu_trsync_timeout=0.25)
        package = types.ModuleType('klippy.extras.danger_options')
        package.get_danger_options = lambda: loaded
        copy = types.ModuleType('extras.danger_options')
        copy.get_danger_options = lambda: stale
        mcu = types.ModuleType('mcu')
        monkeypatch.setitem(sys.modules, 'klippy.extras.danger_options', package)
        monkeypatch.setitem(sys.modules, 'extras.danger_options', copy)
        monkeypatch.setitem(sys.modules, 'mcu', mcu)

        obj, _ = make_afc({'trsync_update': True, 'trsync_timeout': 0.06,
                           'trsync_single_timeout': 0.5})

        assert (loaded.multi_mcu_trsync_timeout, loaded.single_mcu_trsync_timeout) == (0.06, 0.5)
        assert (stale.multi_mcu_trsync_timeout, stale.single_mcu_trsync_timeout) == (0.025, 0.25)
        assert obj.logger.messages == [('info', 'Applying TRSYNC update')]

    def test_top_level_danger_options_raise_timeouts_without_touching_mcu(self, make_afc,
                                                                           monkeypatch):
        # Pre-package Danger Klipper: only the top-level extras module exists.
        options = types.SimpleNamespace(multi_mcu_trsync_timeout=0.01,
                                        single_mcu_trsync_timeout=0.8)
        danger = types.ModuleType('extras.danger_options')
        danger.get_danger_options = lambda: options
        mcu = types.ModuleType('mcu')
        mcu.TRSYNC_TIMEOUT = 0.02
        mcu.TRSYNC_SINGLE_MCU_TIMEOUT = 0.2
        monkeypatch.setitem(sys.modules, 'klippy.extras.danger_options', None)
        monkeypatch.setitem(sys.modules, 'extras.danger_options', danger)
        monkeypatch.setitem(sys.modules, 'mcu', mcu)

        obj, _ = make_afc({'trsync_update': True, 'trsync_timeout': 0.06,
                           'trsync_single_timeout': 0.5})

        assert options.multi_mcu_trsync_timeout == 0.06
        assert options.single_mcu_trsync_timeout == 0.8
        assert mcu.TRSYNC_TIMEOUT == 0.02
        assert mcu.TRSYNC_SINGLE_MCU_TIMEOUT == 0.2
        assert obj.logger.messages == [('info', 'Applying TRSYNC update')]

    def test_ordinary_mcu_fallback_never_reduces_timeouts(self, make_afc, monkeypatch):
        mcu = types.ModuleType('mcu')
        mcu.TRSYNC_TIMEOUT = 0.02
        mcu.TRSYNC_SINGLE_MCU_TIMEOUT = 0.8
        # Klipper: klippy is a plain module and there is no danger_options at all.
        monkeypatch.setitem(sys.modules, 'klippy.extras.danger_options', None)
        monkeypatch.setitem(sys.modules, 'extras.danger_options', None)
        monkeypatch.setitem(sys.modules, 'mcu', mcu)

        obj, _ = make_afc({'trsync_update': True, 'trsync_timeout': 0.06,
                           'trsync_single_timeout': 0.5})

        assert mcu.TRSYNC_TIMEOUT == 0.06
        assert mcu.TRSYNC_SINGLE_MCU_TIMEOUT == 0.8
        assert obj.logger.messages == [('info', 'Applying TRSYNC update')]
