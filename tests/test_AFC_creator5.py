# Unit tests for the Creator 5 toolchanger integration ported from Kalico
# (klippy/extras/AFC*.py at 761a2972; originals in test/test_creator5_afc_sensor.py).
#
# The AFC core, the toolchanger unit, every AFC_extruder and its standalone lane are built
# through their real constructors on one shared MockPrinter. Only hardware is faked: the
# creator5_toolchanger mount pins and C5_TOOL_SELECT macro, gcode_move, toolhead, heaters
# and filament switches.

from __future__ import annotations

import re
import types
from unittest.mock import MagicMock, patch

import pytest
from klippy import message_ready as READY

from extras.AFC import afc as AFC, State
from extras.AFC_extruder import AFCExtruder
from extras.AFC_lane import AFCLaneState
from extras.AFC_Toolchanger import AfcToolchanger
from tests.conftest import MockConfig, MockLogger, MockPrinter


T1_OFFSET = 3.03
T2_OFFSET = 3.2429
PRINT_XY = [100.0, 100.0]
PRINT_Z = 12.0
Z_HOP = 0.4
DOCK_XY = [300.0, 50.0]


class FakeGCodeMove:
    """Klipper gcode_move holding T1's Z offset, nozzle above the last printed layer."""

    def __init__(self) -> None:
        self.base_position = [0.0, 0.0, T1_OFFSET, 0.0]
        self.homing_position = [0.0, 0.0, T1_OFFSET, 0.0]
        self.last_position = PRINT_XY + [PRINT_Z + T1_OFFSET, 50.0]
        self.speed = 25.0
        self.speed_factor = 1.0 / 60.0
        self.absolute_coord = True
        self.absolute_extrude = True
        self.extrude_factor = 1.0
        self.moves: list = []

    def move_with_transform(self, newpos, speed) -> None:
        self.moves.append(list(newpos[:3]))


class FakeToolhead:
    def __init__(self) -> None:
        self.extruder = None
        self.wait_count = 0

    def get_extruder(self):
        return self.extruder

    def flush_step_generation(self) -> None:
        pass

    def set_extruder(self, extruder, extrude_pos) -> None:
        self.extruder = extruder

    def wait_moves(self) -> None:
        self.wait_count += 1

    def get_position(self) -> list:
        return [0.0, 0.0, 0.0, 0.0]

    def get_status(self, eventtime) -> dict:
        return {"axis_maximum": [300.0, 300.0, 300.0, 0.0]}


class FakeHeater:
    def __init__(self) -> None:
        self.target_temp = 0.0
        self.min_extrude_temp = 170.0


class FakeHotend:
    def __init__(self, name: str) -> None:
        self.name = name
        self.heater = FakeHeater()

    def get_heater(self) -> FakeHeater:
        return self.heater


class FakeHeaters:
    def __init__(self) -> None:
        self.calls: list = []

    def set_temperature(self, heater, temp, wait=False) -> None:
        self.calls.append((heater, temp, wait))
        heater.target_temp = temp


class FakeFilamentSwitch:
    def __init__(self) -> None:
        self.runout_helper = types.SimpleNamespace(sensor_enabled=True)


class FakeCreator5Board:
    """creator5_toolchanger: dock/grab pins, and what C5_TOOL_SELECT does to the machine."""

    def __init__(self, gcode_move: FakeGCodeMove, offsets: dict) -> None:
        self.gcode_move = gcode_move
        self.offsets = offsets
        self.attached = None
        self.fault = None
        self.select_fault = None
        self.attach_on_select = True
        self.travel_z = None
        self.pin_reads = 0

    def attached_tool_from_pins(self):
        self.pin_reads += 1
        if self.fault is not None:
            raise RuntimeError(self.fault)
        return self.attached

    def select(self, index: int) -> None:
        if self.select_fault is not None:
            raise RuntimeError(self.select_fault)
        delta = self.offsets[index] - self.gcode_move.homing_position[2]
        self.gcode_move.base_position[2] += delta
        self.gcode_move.homing_position[2] += delta
        self.gcode_move.last_position[:2] = DOCK_XY
        if self.travel_z is not None:
            self.gcode_move.last_position[2] = self.travel_z
        if self.attach_on_select:
            self.attached = index


class Creator5Rig:
    def __init__(self) -> None:
        self.printer = MockPrinter()
        # A MockLogger for construction avoids AFC_logger's root-logger and atexit side effects.
        # It stays with afcDeltaTime, keeping timing lines out of the asserted logger.
        self.timing_logger = MockLogger()
        with patch("extras.AFC.AFC_logger", return_value=self.timing_logger), \
             patch.object(AFC, "print_version"):
            self.afc = AFC(MockConfig("AFC", self.printer, {"z_hop": Z_HOP}))
        self.printer._afc = self.afc
        self.logger = MockLogger()
        self.afc.logger = self.logger
        # Moonraker-backed statistics
        self.afc.afc_stats = MagicMock()
        self.gcode_move = FakeGCodeMove()
        self.toolhead = FakeToolhead()
        self.afc.gcode_move = self.gcode_move
        self.afc.toolhead = self.toolhead
        function = self.afc.function
        function.is_homed.return_value = True
        function.is_paused.return_value = False
        function.is_printing.return_value = False
        function.get_current_extruder_obj.return_value = None
        function.get_current_lane.return_value = None

        self.board = FakeCreator5Board(self.gcode_move, {1: T1_OFFSET, 2: T2_OFFSET})
        self.printer._objects["creator5_toolchanger"] = self.board
        self.heaters = FakeHeaters()
        self.printer._objects["heaters"] = self.heaters
        self.scripts: list = []
        self.console: list = []
        gcode = self.printer._gcode
        gcode.run_script_from_command.side_effect = self._run_script
        gcode.run_script = MagicMock()
        gcode.respond_info = self.console.append

        self.unit = AfcToolchanger(MockConfig("AFC_Toolchanger Tools", self.printer))

    def _run_script(self, script: str) -> None:
        self.scripts.append(script)
        match = re.fullmatch(r"C5_TOOL_SELECT T=(\d+)", script)
        if match:
            self.board.select(int(match.group(1)))

    def add_head(self, name: str, index=None, tcmd: str = "T0", values=None,
                 standalone: bool = True) -> AFCExtruder:
        config_values = {"map": [tcmd]}
        if standalone:
            config_values["toolchanger_unit"] = "Tools"
        if index is not None:
            config_values["creator5_tool_index"] = index
            config_values["custom_tool_swap"] = f"C5_TOOL_SELECT T={index}"
        config_values.update(values or {})
        head = AFCExtruder(MockConfig(f"AFC_extruder {name}", self.printer, config_values))
        # Connect-time links, with hardware and Moonraker faked
        head.estats = MagicMock()
        head.toolhead_extruder = FakeHotend(name)
        head.tc_unit_obj = self.unit
        lane = head.tc_lane
        if lane is not None:
            lane.unit_obj = MagicMock()
            lane.extruder_obj = head
            head.lanes[lane.name] = lane
            self.afc.lanes[lane.name] = lane
        self.afc.tools[name] = head
        head.handle_ready()
        self.logger.messages.clear()
        return head


@pytest.fixture
def rig():
    built = Creator5Rig()
    yield built
    built.afc.join_threads()


def _non_debug(rig: Creator5Rig) -> list:
    return [entry for entry in rig.logger.messages if entry[0] != "debug"]


def _approx_moves(moves: list):
    return [pytest.approx(move) for move in moves]


def _mounted_t1_with_docked_t2(rig: Creator5Rig, travel_z=None):
    """T1 printing with its own filament, T2 docked holding filament."""
    t1 = rig.add_head("extruder1", index=1, tcmd="T1")
    t2 = rig.add_head("extruder2", index=2, tcmd="T2")
    rig.board.attached = 1
    rig.board.travel_z = travel_z
    rig.toolhead.extruder = t1.toolhead_extruder
    t1.tool_start_state = True
    t1.tc_lane.set_tool_loaded()
    t2.tool_start_state = True
    t2.tc_lane.set_tool_loaded()
    t2.on_shuttle()
    assert t2.tc_lane.status == AFCLaneState.LOADED
    rig.logger.messages.clear()
    return t1, t2


# on_shuttle

class TestOnShuttle:
    def test_mounting_a_filled_head_marks_its_lane_loaded(self, rig):
        head = rig.add_head("extruder1", index=1, tcmd="T1")
        head.tool_start_state = True
        rig.board.attached = 1

        assert head.on_shuttle() is True
        assert head.tc_lane.tool_loaded is True
        assert head.tc_lane.status == AFCLaneState.LOADED
        assert head.lane_loaded == "extruder1"
        assert head.creator5_mount_error is None

    def test_restored_tool_loaded_without_status_is_resynced(self, rig):
        # AFC_prep restores tool_loaded after a restart but leaves status at NONE
        head = rig.add_head("extruder1", index=1, tcmd="T1")
        head.tool_start_state = True
        head.tc_lane.tool_loaded = True
        rig.board.attached = 1

        assert head.on_shuttle() is True
        assert head.tc_lane.status == AFCLaneState.LOADED
        head.tc_lane.unit_obj.lane_loaded.assert_called_once_with(head.tc_lane)
        head.tc_lane.unit_obj.lane_tool_loaded.assert_not_called()

    def test_docking_keeps_filament_but_leaves_tooled(self, rig):
        head = rig.add_head("extruder1", index=1, tcmd="T1")
        head.tool_start_state = True
        head.tc_lane.set_tool_loaded()
        rig.board.attached = 2

        assert head.on_shuttle() is False
        assert head.tc_lane.status == AFCLaneState.LOADED
        assert head.tc_lane.tool_loaded is True
        assert head.lane_loaded == "extruder1"

    def test_docked_poll_does_not_reannounce_a_loaded_lane(self, rig):
        head = rig.add_head("extruder1", index=1, tcmd="T1")
        head.tc_lane.tool_loaded = True
        head.tc_lane.status = AFCLaneState.LOADED

        assert head.on_shuttle() is False
        head.tc_lane.unit_obj.lane_loaded.assert_not_called()

    def test_docked_head_with_unloaded_lane_stays_unloaded(self, rig):
        head = rig.add_head("extruder1", index=1, tcmd="T1")
        head.tool_start_state = True

        assert head.on_shuttle() is False
        assert head.tc_lane.status == AFCLaneState.NONE
        assert head.tc_lane.tool_loaded is False

    def test_mounted_head_without_filament_is_not_marked_loaded(self, rig):
        head = rig.add_head("extruder1", index=1, tcmd="T1")
        rig.board.attached = 1

        assert head.on_shuttle() is True
        assert head.tc_lane.status == AFCLaneState.NONE
        assert head.tc_lane.tool_loaded is False

    @pytest.mark.parametrize("busy", [AFCLaneState.TOOL_LOADING, AFCLaneState.TOOL_UNLOADING,
                                      AFCLaneState.EJECTING, AFCLaneState.CALIBRATING,
                                      AFCLaneState.INFINITE_RUNOUT, AFCLaneState.ERROR])
    def test_busy_lane_is_left_alone(self, rig, busy):
        head = rig.add_head("extruder1", index=1, tcmd="T1")
        head.tool_start_state = True
        head.tc_lane.status = busy
        rig.board.attached = 1

        assert head.on_shuttle() is True
        assert head.tc_lane.status == busy
        assert head.tc_lane.tool_loaded is False

    def test_another_attached_head_means_not_mounted(self, rig):
        head = rig.add_head("extruder1", index=1, tcmd="T1")
        head.tool_start_state = True
        rig.board.attached = 2

        assert head.on_shuttle() is False
        assert head.tc_lane.status == AFCLaneState.NONE

    def test_unit_fed_head_leaves_its_own_lane_alone(self, rig):
        head = rig.add_head("extruder1", index=1, tcmd="T1")
        head.no_lanes = False
        head.tool_start_state = True
        rig.board.attached = 1

        assert head.on_shuttle() is True
        assert head.tc_lane.status == AFCLaneState.NONE
        assert head.tc_lane.tool_loaded is False

    def test_contradictory_mount_switches_report_unmounted_with_error(self, rig):
        head = rig.add_head("extruder1", index=1, tcmd="T1")
        head.tc_lane.set_tool_loaded()
        rig.board.attached = 1
        rig.board.fault = "T1 dock and grab switches disagree"

        assert head.on_shuttle() is False
        assert head.creator5_mount_error == "T1 dock and grab switches disagree"
        assert head.tc_lane.status == AFCLaneState.TOOLED

    def test_valid_read_clears_previous_mount_error(self, rig):
        head = rig.add_head("extruder1", index=1, tcmd="T1")
        rig.board.fault = "T1 dock and grab switches disagree"
        head.on_shuttle()
        rig.board.fault = None
        rig.board.attached = 1

        assert head.on_shuttle() is True
        assert head.creator5_mount_error is None

    def test_head_without_standalone_lane_reports_mount_only(self, rig):
        head = rig.add_head("extruder1", index=1, tcmd="T1", standalone=False)
        rig.board.attached = 1

        assert head.tc_lane is None
        assert head.on_shuttle() is True

    def test_generic_head_never_reads_creator5_mount_pins(self, rig):
        head = rig.add_head("extruder3", tcmd="T3")
        rig.board.fault = "T1 dock and grab switches disagree"

        assert head.on_shuttle() is True
        assert rig.board.pin_reads == 0
        assert head.creator5_mount_error is None


# get_status

class TestGetStatus:
    @pytest.mark.parametrize(("attached", "printing", "head_status", "expected"), [
        (None, True, State.IDLE, State.PARKED),
        (1, True, State.IDLE, State.PRINTING),
        (1, False, State.IDLE, State.IDLE),
        (1, True, State.TOOL_PICKUP, State.TOOL_PICKUP),
        (1, True, State.TOOL_DOCK, State.TOOL_DOCK),
        (1, False, State.ERROR, State.ERROR),
        (None, False, State.ERROR, State.ERROR),
    ])
    def test_reports_physical_head_state(self, rig, attached, printing, head_status,
                                         expected):
        head = rig.add_head("extruder1", index=1, tcmd="T1")
        head.status = head_status
        rig.board.attached = attached
        rig.afc.function.is_printing.return_value = printing

        status = head.get_status()

        assert status["status"] == expected
        assert status["on_shuttle"] is (attached == 1)
        assert status["mount_sensor_error"] is None

    def test_invalid_mount_sensors_report_error(self, rig):
        head = rig.add_head("extruder1", index=1, tcmd="T1")
        rig.board.attached = 1
        rig.board.fault = "T1 dock and grab switches disagree"

        status = head.get_status()

        assert status["status"] == State.ERROR
        assert status["mount_sensor_error"] == "T1 dock and grab switches disagree"
        assert status["on_shuttle"] is False

    def test_generic_head_reports_its_own_status(self, rig):
        head = rig.add_head("extruder3", tcmd="T3")
        rig.board.fault = "T1 dock and grab switches disagree"
        rig.afc.function.is_printing.return_value = True

        status = head.get_status()

        assert status["status"] == State.IDLE
        assert "mount_sensor_error" not in status
        assert status["on_shuttle"] is True
        assert rig.board.pin_reads == 0


# prep_on_shuttle_check

class TestPrepOnShuttleCheck:
    @pytest.mark.parametrize(("attached", "expected"), [
        (1, "<span class=success--text> mounted</span>"),
        (None, "<span class=primary--text> parked</span>"),
        (2, "<span class=primary--text> parked</span>"),
    ])
    def test_reports_mount_state(self, rig, attached, expected):
        head = rig.add_head("extruder1", index=1, tcmd="T1")
        rig.board.attached = attached

        assert head.prep_on_shuttle_check(head.tc_lane) == expected

    def test_invalid_mount_sensors_are_reported(self, rig):
        head = rig.add_head("extruder1", index=1, tcmd="T1")
        rig.board.fault = "T1 dock and grab switches disagree"

        assert (head.prep_on_shuttle_check(head.tc_lane)
                == "<span class=error--text> mount sensors invalid</span>")
        assert head.creator5_mount_error == "T1 dock and grab switches disagree"

    def test_generic_head_reports_filament_in_toolhead(self, rig):
        head = rig.add_head("extruder3", tcmd="T3")

        assert (head.prep_on_shuttle_check(head.tc_lane)
                == "<span class=primary--text> in ToolHead</span>")
        assert rig.board.pin_reads == 0


# AfcToolchanger.system_Test (PREP line)

class TestSystemTest:
    def test_docked_head_with_filament(self, rig):
        head = rig.add_head("extruder1", index=1, tcmd="T1")
        head.tool_start_state = True

        assert rig.unit.system_Test(head.tc_lane, 0.0, "", False) is True

        assert rig.logger.messages == [("raw", "extruder1 tool cmd: T1  "
                                        "<span class=success--text>FILAMENT PRESENT</span>"
                                        "<span class=primary--text> parked</span>")]
        assert head.tc_lane._afc_prep_done is True

    def test_mounted_empty_head_ignores_lane_load_flags(self, rig):
        head = rig.add_head("extruder1", index=1, tcmd="T1")
        head.tc_lane.prep_state = True
        head.tc_lane._load_state = True
        head.lane_loaded = "extruder1"
        rig.board.attached = 1

        rig.unit.system_Test(head.tc_lane, 0.0, "", False)

        assert rig.logger.messages == [("raw", "extruder1 tool cmd: T1  "
                                        "<span class=primary--text>FILAMENT ABSENT</span>"
                                        "<span class=success--text> mounted</span>")]

    def test_invalid_mount_sensors(self, rig):
        head = rig.add_head("extruder1", index=1, tcmd="T1")
        rig.board.fault = "T1 dock and grab switches disagree"

        rig.unit.system_Test(head.tc_lane, 0.0, "", False)

        assert rig.logger.messages == [("raw", "extruder1 tool cmd: T1  "
                                        "<span class=primary--text>FILAMENT ABSENT</span>"
                                        "<span class=error--text> mount sensors invalid</span>")]

    def test_generic_loaded_head_keeps_loaded_label(self, rig):
        head = rig.add_head("extruder3", tcmd="T3")
        head.tc_lane.prep_state = True
        head.tc_lane._load_state = True
        head.lane_loaded = "extruder3"

        rig.unit.system_Test(head.tc_lane, 0.0, "", False)

        assert rig.logger.messages == [("raw", "extruder3 tool cmd: T3  "
                                        "<span class=success--text>LOADED</span>"
                                        "<span class=primary--text> in ToolHead</span>")]
        assert rig.board.pin_reads == 0

    def test_generic_unloaded_head_has_no_label(self, rig):
        head = rig.add_head("extruder3", tcmd="T3")
        head.tool_start_state = True

        rig.unit.system_Test(head.tc_lane, 0.0, "", False)

        assert rig.logger.messages == [("raw", "extruder3 tool cmd: T3  ")]


# tool_start_callback with auto_load_on_tool_start disabled

class TestToolStartCallback:
    def _head(self, rig):
        head = rig.add_head("extruder1", index=1, tcmd="T1",
                            values={"auto_load_on_tool_start": False})
        head.tc_lane._afc_prep_done = True
        rig.printer.state_message = READY
        return head

    def test_filament_fed_into_parked_head_loads_once_mounted(self, rig):
        head = self._head(rig)

        head.tool_start_callback(0.0, True)

        assert head.tool_start_state is True
        assert head.tc_lane.prep_state is True
        assert head.tc_lane.status == AFCLaneState.NONE
        assert head.tc_lane.tool_loaded is False
        assert head.load_active is False

        rig.board.attached = 1
        assert head.on_shuttle() is True
        assert head.tc_lane.status == AFCLaneState.LOADED
        assert head.tc_lane.tool_loaded is True
        assert rig.logger.messages == []

    def test_filament_fed_into_mounted_head_marks_loaded_without_extruding(self, rig):
        head = self._head(rig)
        rig.board.attached = 1

        head.tool_start_callback(0.0, True)

        assert head.tc_lane.status == AFCLaneState.LOADED
        assert head.tc_lane.tool_loaded is True
        assert head.lane_loaded == "extruder1"
        assert head.load_active is False
        assert head.current_move_distance == 0
        assert rig.logger.messages == []

    def test_filament_removed_unloads_lane(self, rig):
        head = self._head(rig)
        rig.board.attached = 1
        head.tool_start_callback(0.0, True)

        head.tool_start_callback(1.0, False)

        assert head.tool_start_state is False
        assert head.tc_lane.status == AFCLaneState.NONE
        assert head.tc_lane.tool_loaded is False
        assert head.lane_loaded is None


# load_unload_sequence (manual unload)

class TestLoadUnloadSequence:
    @pytest.mark.parametrize("auto_load_unload", [True, False])
    def test_creator5_unload_asks_for_the_cutter(self, rig, auto_load_unload):
        head = rig.add_head("extruder1", index=1, tcmd="T1",
                            values={"standalone_auto_load_unload": auto_load_unload})
        head.tc_lane.set_tool_loaded()

        head.load_unload_sequence(-100)

        assert rig.console == ["T1: use the cutter to remove the filament; it does not "
                               "retract automatically."]
        assert head.tc_lane.status == AFCLaneState.TOOLED
        assert head.tc_lane.tool_loaded is True
        assert head.lane_loaded == "extruder1"
        assert head.load_active is False
        assert head.current_move_distance == 0
        assert rig.logger.messages == []

    def test_creator5_load_is_not_guarded(self, rig):
        head = rig.add_head("extruder1", index=1, tcmd="T1",
                            values={"standalone_auto_load_unload": False})

        head.load_unload_sequence(100)

        # Marked loaded; parked, so on_shuttle() leaves it LOADED rather than TOOLED
        assert rig.console == []
        assert head.tc_lane.status == AFCLaneState.LOADED
        assert head.tc_lane.tool_loaded is True
        assert head.lane_loaded == "extruder1"
        head.tc_lane.unit_obj.lane_tool_loaded_idle.assert_called_once_with(head.tc_lane)
        assert rig.logger.messages == [
            ("info", "Loading extruder1 skipped: standalone_auto_load_unload is disabled, "
                     "hotend heating and filament move were not performed")]

    def test_generic_head_unloads(self, rig):
        head = rig.add_head("extruder3", tcmd="T3",
                            values={"standalone_auto_load_unload": False})
        head.tc_lane.set_tool_loaded()

        head.load_unload_sequence(-100)

        assert rig.console == []
        assert head.tc_lane.status == AFCLaneState.NONE
        assert head.tc_lane.tool_loaded is False
        assert head.lane_loaded is None
        assert rig.logger.messages == [
            ("info", "Unloading extruder3 skipped: standalone_auto_load_unload is disabled, "
                     "hotend heating and filament move were not performed")]


# AfcToolchanger.tool_swap

class TestToolSwap:
    def test_creator5_swap_activates_once_without_waiting(self, rig):
        _, t2 = _mounted_t1_with_docked_t2(rig)

        rig.unit.tool_swap(t2.tc_lane)

        assert rig.scripts == ["C5_TOOL_SELECT T=2"]
        assert rig.board.attached == 2
        assert rig.toolhead.extruder is t2.toolhead_extruder
        assert rig.toolhead.wait_count == 0
        rig.afc.function._handle_activate_extruder.assert_not_called()
        assert rig.afc.homing_position[2] == pytest.approx(T2_OFFSET)
        assert rig.afc.base_position[2] == pytest.approx(T2_OFFSET)
        assert t2.status == State.IDLE
        assert t2.next_pickup is False
        assert _non_debug(rig) == [("info", "Running custom select: C5_TOOL_SELECT T=2")]

    def test_generic_swap_activates_again_and_waits(self, rig):
        head = rig.add_head("extruder3", tcmd="T3")

        rig.unit.tool_swap(head.tc_lane)

        assert rig.scripts == ["SELECT_TOOL T=3"]
        assert rig.toolhead.extruder is head.toolhead_extruder
        assert rig.toolhead.wait_count == 1
        rig.afc.function._handle_activate_extruder.assert_called_once_with(0)

    @pytest.mark.parametrize(("total", "current", "error_state", "expected"), [
        (3, 1, False, 2),
        (0, 0, False, 0),
        (3, 3, False, 3),
        (3, 1, True, 1),
    ])
    def test_creator5_swap_advances_print_toolchange_count(self, rig, total, current,
                                                          error_state, expected):
        _, t2 = _mounted_t1_with_docked_t2(rig)
        rig.afc.number_of_toolchanges = total
        rig.afc.current_toolchange = current
        rig.afc.error_state = error_state

        rig.unit.tool_swap(t2.tc_lane)

        assert rig.afc.current_toolchange == expected

    def test_generic_swap_leaves_count_to_change_tool(self, rig):
        head = rig.add_head("extruder3", tcmd="T3")
        rig.afc.number_of_toolchanges = 3
        rig.afc.current_toolchange = 1

        rig.unit.tool_swap(head.tc_lane)

        assert rig.afc.current_toolchange == 1


# CHANGE_TOOL

class TestChangeTool:
    def test_swaps_head_only_and_lifts_before_returning_over_the_print(self, rig):
        t1, t2 = _mounted_t1_with_docked_t2(rig)
        lift = PRINT_Z + T2_OFFSET + Z_HOP
        final = PRINT_Z + T2_OFFSET

        rig.afc.CHANGE_TOOL(t2.tc_lane)

        assert rig.scripts == ["C5_TOOL_SELECT T=2"]
        assert rig.board.attached == 2
        assert rig.toolhead.extruder is t2.toolhead_extruder
        assert rig.gcode_move.moves == _approx_moves([DOCK_XY + [lift], PRINT_XY + [lift],
                                                      PRINT_XY + [final]])
        assert rig.gcode_move.last_position[:3] == pytest.approx(PRINT_XY + [final])
        assert rig.gcode_move.homing_position[2] == pytest.approx(T2_OFFSET)
        assert rig.gcode_move.base_position[2] == pytest.approx(T2_OFFSET)
        assert rig.afc.position_saved is False
        assert rig.afc.in_toolchange is False
        # Filament stays where it is: no unload of T1, no load of T2
        assert t1.tc_lane.status == AFCLaneState.TOOLED
        assert t1.tc_lane.tool_loaded is True
        assert t2.tc_lane.status == AFCLaneState.LOADED
        assert t2.tc_lane.tool_loaded is True
        assert _non_debug(rig) == [("info", "Running custom select: C5_TOOL_SELECT T=2")]

    def test_keeps_a_higher_travel_clearance_for_the_return(self, rig):
        _, t2 = _mounted_t1_with_docked_t2(rig, travel_z=20.0)

        rig.afc.CHANGE_TOOL(t2.tc_lane)

        assert rig.gcode_move.moves == _approx_moves([PRINT_XY + [20.0],
                                                      PRINT_XY + [PRINT_Z + T2_OFFSET]])

    def test_without_restore_leaves_saved_position_in_new_head_frame(self, rig):
        _, t2 = _mounted_t1_with_docked_t2(rig)

        rig.afc.CHANGE_TOOL(t2.tc_lane, restore_pos=False)

        assert rig.gcode_move.moves == []
        assert rig.afc.position_saved is True
        assert rig.afc.last_gcode_position[:3] == pytest.approx(PRINT_XY
                                                                + [PRINT_Z + T2_OFFSET])
        assert rig.afc.in_toolchange is False

    def test_success_converts_all_saved_axes_once(self, rig):
        _, t2 = _mounted_t1_with_docked_t2(rig)
        rig.gcode_move.base_position[:2] = [2.5, -1.25]
        rig.gcode_move.homing_position[:2] = [2.5, -1.25]
        rig.gcode_move.last_position[:2] = [102.5, 98.75]
        select = rig.board.select

        def select_with_xy_offsets(index):
            select(index)
            rig.gcode_move.base_position[:2] = [-1.0, 4.0]
            rig.gcode_move.homing_position[:2] = [-1.0, 4.0]

        rig.board.select = select_with_xy_offsets
        rig.afc.CHANGE_TOOL(t2.tc_lane, restore_pos=False)

        assert rig.afc.position_saved is True
        assert rig.afc.last_gcode_position[:3] == pytest.approx(
            [99.0, 104.0, PRINT_Z + T2_OFFSET])
        rig.afc.restore_pos(False, keep_offsets=True)
        assert rig.gcode_move.moves[-1] == pytest.approx(
            [99.0, 104.0, PRINT_Z + T2_OFFSET])


    def test_position_saved_by_caller_is_not_restored_here(self, rig):
        _, t2 = _mounted_t1_with_docked_t2(rig)
        rig.afc.save_pos()

        rig.afc.CHANGE_TOOL(t2.tc_lane)

        assert rig.scripts == ["C5_TOOL_SELECT T=2"]
        assert rig.gcode_move.moves == []
        assert rig.afc.position_saved is True

    def test_nested_change_keeps_outer_toolchange_flag(self, rig):
        _, t2 = _mounted_t1_with_docked_t2(rig)
        rig.afc.in_toolchange = True

        rig.afc.CHANGE_TOOL(t2.tc_lane)

        assert rig.afc.in_toolchange is True
        assert rig.afc.position_saved is False
        assert rig.gcode_move.moves == []

    def test_failed_grab_raises_without_returning_to_the_print(self, rig):
        _, t2 = _mounted_t1_with_docked_t2(rig)
        rig.board.select_fault = "T2 grab switch not confirmed"

        with pytest.raises(RuntimeError, match="T2 grab switch not confirmed"):
            rig.afc.CHANGE_TOOL(t2.tc_lane)

        assert rig.board.attached == 1
        assert rig.gcode_move.moves == []
        assert rig.afc.in_toolchange is False
        rig.afc.error.AFC_error.assert_not_called()

    def test_failed_grab_preserves_saved_frame_for_later_restore(self, rig):
        _, t2 = _mounted_t1_with_docked_t2(rig)
        rig.gcode_move.base_position[:2] = [2.5, -1.25]
        rig.gcode_move.homing_position[:2] = [2.5, -1.25]
        rig.gcode_move.last_position[:2] = [102.5, 98.75]
        saved_position = list(rig.gcode_move.last_position)
        saved_base = list(rig.gcode_move.base_position)
        saved_homing = list(rig.gcode_move.homing_position)
        rig.board.select_fault = "T2 grab switch not confirmed"

        with pytest.raises(RuntimeError, match="T2 grab switch not confirmed"):
            rig.afc.CHANGE_TOOL(t2.tc_lane)

        assert rig.afc.position_saved is True
        assert rig.afc.last_gcode_position == pytest.approx(saved_position)
        assert rig.afc.base_position == pytest.approx(saved_base)
        assert rig.afc.homing_position == pytest.approx(saved_homing)
        rig.gcode_move.last_position[:3] = DOCK_XY + [20.0]
        rig.afc.restore_pos(False)
        assert rig.gcode_move.moves[-1] == pytest.approx(saved_position[:3])
        assert rig.afc.position_saved is False


    def test_creator5_head_is_swapped_even_when_already_current(self, rig):
        t1, _ = _mounted_t1_with_docked_t2(rig)
        rig.afc.function.get_current_lane.return_value = "extruder1"

        rig.afc.CHANGE_TOOL(t1.tc_lane)

        assert rig.scripts == ["C5_TOOL_SELECT T=1"]
        assert _non_debug(rig) == [("info", "Running custom select: C5_TOOL_SELECT T=1")]

    def test_generic_current_lane_is_not_swapped(self, rig):
        head = rig.add_head("extruder3", tcmd="T3")
        rig.afc.function.get_current_lane.return_value = "extruder3"

        rig.afc.CHANGE_TOOL(head.tc_lane)

        assert rig.scripts == []
        rig.afc.function._handle_activate_extruder.assert_called_once_with(0)
        assert _non_debug(rig) == [("info", "extruder3 already loaded")]


# restore_pos

class TestRestorePos:
    def _saved_then_t2_selected(self, rig, travel_z=None):
        rig.afc.save_pos()
        rig.board.travel_z = travel_z
        rig.board.select(2)

    def test_keep_offsets_lifts_before_xy_when_new_nozzle_is_lower(self, rig):
        self._saved_then_t2_selected(rig)
        lift = PRINT_Z + T2_OFFSET + Z_HOP
        final = PRINT_Z + T2_OFFSET

        rig.afc.restore_pos(move_z_first=False, keep_offsets=True)

        assert rig.gcode_move.moves == _approx_moves([DOCK_XY + [lift], PRINT_XY + [lift],
                                                      PRINT_XY + [final]])
        assert rig.gcode_move.homing_position[2] == pytest.approx(T2_OFFSET)
        assert rig.gcode_move.base_position[2] == pytest.approx(T2_OFFSET)
        assert rig.afc.position_saved is False
        assert rig.afc.current_state == State.IDLE

    def test_keep_offsets_keeps_higher_clearance(self, rig):
        self._saved_then_t2_selected(rig, travel_z=20.0)

        rig.afc.restore_pos(move_z_first=False, keep_offsets=True)

        assert rig.gcode_move.moves == _approx_moves([PRINT_XY + [20.0],
                                                      PRINT_XY + [PRINT_Z + T2_OFFSET]])

    def test_keep_offsets_with_z_first_moves_to_hop_height_first(self, rig):
        self._saved_then_t2_selected(rig, travel_z=20.0)
        lift = PRINT_Z + T2_OFFSET + Z_HOP

        rig.afc.restore_pos(move_z_first=True, keep_offsets=True)

        assert rig.gcode_move.moves == _approx_moves([DOCK_XY + [lift], PRINT_XY + [lift],
                                                      PRINT_XY + [PRINT_Z + T2_OFFSET]])

    def test_default_restore_returns_saved_offsets_without_lift(self, rig):
        self._saved_then_t2_selected(rig)
        saved_z = PRINT_Z + T1_OFFSET

        rig.afc.restore_pos(move_z_first=False)

        assert rig.gcode_move.moves == _approx_moves([PRINT_XY + [saved_z],
                                                      PRINT_XY + [saved_z]])
        assert rig.gcode_move.homing_position[2] == pytest.approx(T1_OFFSET)
        assert rig.gcode_move.base_position[2] == pytest.approx(T1_OFFSET)

    def test_nothing_saved_means_no_moves(self, rig):
        rig.board.select(2)

        rig.afc.restore_pos(move_z_first=False, keep_offsets=True)

        assert rig.gcode_move.moves == []
        assert rig.gcode_move.homing_position[2] == pytest.approx(T2_OFFSET)


# AFCLane.handle_toolhead_runout -> _creator5_runout

class TestHandleToolheadRunout:
    def _printing_t1(self, rig, runout_lane="extruder2", spare_loaded=True):
        t1 = rig.add_head("extruder1", index=1, tcmd="T1")
        spare = rig.add_head("extruder2", index=2, tcmd="T2")
        spare.tool_start_state = spare_loaded
        t1.fila_tool_start = FakeFilamentSwitch()
        t1.tc_lane.set_tool_loaded()
        t1.tc_lane.runout_lane = runout_lane
        rig.board.attached = 1
        rig.afc.function.is_printing.return_value = True
        rig.logger.messages.clear()
        return t1

    def test_runout_with_loaded_spare_switches_without_asking(self, rig):
        t1 = self._printing_t1(rig)
        events = []

        def pause():
            events.append("pause")
            rig.gcode_move.last_position[:3] = [110.0, 120.0, PRINT_Z + T1_OFFSET]

        rig.afc.error.pause_resume.send_pause_command.side_effect = pause
        # Stale position from an earlier change must not be reused
        rig.afc.position_saved = True
        rig.afc.last_gcode_position = [1.0, 2.0, 3.0, 0.0]

        t1.tc_lane.handle_toolhead_runout(sensor="tool_start")

        assert events == ["pause"]
        assert rig.afc.position_saved is True
        assert rig.afc.last_gcode_position[:3] == [110.0, 120.0, PRINT_Z + T1_OFFSET]
        rig.printer._gcode.run_script.assert_called_once_with(
            "PAUSE\nC5_FILAMENT_RUNOUT TOOL=1 TO=2")
        assert t1.tc_lane.status == AFCLaneState.NONE
        assert t1.tc_lane.tool_loaded is False
        assert t1.lane_loaded is None
        t1.tc_lane.unit_obj.lane_not_ready.assert_called_with(t1.tc_lane)
        assert _non_debug(rig) == [("info", "Infinite Spool: T1 ran out, switching to T2")]
        assert rig.scripts == []

    def test_runout_without_spare_asks(self, rig):
        t1 = self._printing_t1(rig, runout_lane=None)

        t1.tc_lane.handle_toolhead_runout(sensor="tool_start")

        rig.printer._gcode.run_script.assert_called_once_with("PAUSE\nC5_FILAMENT_RUNOUT TOOL=1")
        assert _non_debug(rig) == [("info", "T1 ran out of filament")]

    def test_runout_with_empty_spare_asks(self, rig):
        t1 = self._printing_t1(rig, spare_loaded=False)

        t1.tc_lane.handle_toolhead_runout(sensor="tool_start")

        rig.printer._gcode.run_script.assert_called_once_with("PAUSE\nC5_FILAMENT_RUNOUT TOOL=1")
        assert _non_debug(rig) == [
            ("error", "Runout lane extruder2 has no filament; asking instead"),
            ("info", "T1 ran out of filament")]

    @pytest.mark.parametrize("runout_lane", ["extruder1", "extruder3", "extruder9"])
    def test_runout_lane_that_is_not_another_creator5_head_asks(self, rig, runout_lane):
        rig.add_head("extruder3", tcmd="T3")
        t1 = self._printing_t1(rig, runout_lane=runout_lane)

        t1.tc_lane.handle_toolhead_runout(sensor="tool_start")

        rig.printer._gcode.run_script.assert_called_once_with("PAUSE\nC5_FILAMENT_RUNOUT TOOL=1")
        assert _non_debug(rig) == [
            ("error", f"Runout lane {runout_lane} for extruder1 is not another Creator 5 tool"),
            ("info", "T1 ran out of filament")]

    def test_failed_runout_macro_raises_afc_error(self, rig):
        t1 = self._printing_t1(rig)
        rig.printer._gcode.run_script.side_effect = RuntimeError("Unknown command")

        t1.tc_lane.handle_toolhead_runout(sensor="tool_start")

        rig.afc.error.AFC_error.assert_called_once_with(
            "Creator 5 runout handling failed: Unknown command")

    def test_generic_standalone_head_uses_upstream_infinite_spool(self, rig):
        head = rig.add_head("extruder3", tcmd="T3")
        head.fila_tool_start = FakeFilamentSwitch()
        head.tc_lane.set_tool_loaded()
        head.tc_lane.runout_lane = "extruder4"
        rig.afc.function.is_printing.return_value = True
        rig.logger.messages.clear()

        head.tc_lane.handle_toolhead_runout(sensor="tool_start")

        rig.afc.error.AFC_error.assert_called_once_with(
            "Error when looking up current lane:None")
        rig.printer._gcode.run_script.assert_not_called()
        assert rig.board.pin_reads == 0
        assert head.tc_lane.status == AFCLaneState.NONE
        assert _non_debug(rig) == [("info", "Infinite Spool triggered for extruder3")]


# AFCLane._perform_infinite_runout -> _perform_creator5_infinite_runout

class TestPerformInfiniteRunout:
    def _runout(self, rig):
        t1, t2 = _mounted_t1_with_docked_t2(rig)
        t1.toolhead_extruder.heater.target_temp = 220.0
        t1.tc_lane.runout_lane = "extruder2"
        return t1, t2

    def test_switches_to_loaded_spare_head_and_resumes(self, rig):
        t1, t2 = self._runout(rig)
        lift = PRINT_Z + T2_OFFSET + Z_HOP
        final = PRINT_Z + T2_OFFSET

        t1.tc_lane._perform_infinite_runout()

        assert rig.heaters.calls == [(t2.toolhead_extruder.heater, 220.0, True)]
        assert rig.scripts == ["C5_TOOL_SELECT T=2",
                               "AFC_SWAP_MAPPING FROM=extruder1 TO=extruder2"]
        assert rig.board.attached == 2
        assert rig.toolhead.extruder is t2.toolhead_extruder
        assert t1.tc_lane.status == AFCLaneState.NONE
        assert t1.tc_lane.tool_loaded is False
        assert t1.lane_loaded is None
        assert t2.tc_lane.status == AFCLaneState.TOOLED
        assert t2.tc_lane.tool_loaded is True
        assert t2.lane_loaded == "extruder2"
        assert rig.gcode_move.moves == _approx_moves([DOCK_XY + [lift], PRINT_XY + [lift],
                                                      PRINT_XY + [final]])
        assert rig.gcode_move.homing_position[2] == pytest.approx(T2_OFFSET)
        assert rig.afc.position_saved is False
        pause_resume = rig.afc.error.pause_resume
        pause_resume.send_pause_command.assert_called_once_with()
        pause_resume.send_resume_command.assert_called_once_with()
        rig.afc.error.AFC_error.assert_not_called()
        assert _non_debug(rig) == [("info", "Running custom select: C5_TOOL_SELECT T=2"),
                                   ("info", "Infinite Spool switched extruder1 to extruder2")]

    def test_empty_spare_keeps_print_paused_without_moving(self, rig):
        t1, t2 = self._runout(rig)
        t2.tool_start_state = False

        t1.tc_lane._perform_infinite_runout()

        rig.afc.error.AFC_error.assert_called_once_with(
            "Infinite-spool replacement extruder2 has no filament; print paused")
        rig.afc.error.pause_resume.send_pause_command.assert_not_called()
        assert rig.scripts == []
        assert rig.board.attached == 1
        assert t1.tc_lane.status == AFCLaneState.TOOLED

    @pytest.mark.parametrize("runout_lane", [None, "extruder1", "extruder3"])
    def test_invalid_spare_is_refused(self, rig, runout_lane):
        t1, _ = self._runout(rig)
        rig.add_head("extruder3", tcmd="T3")
        t1.tc_lane.runout_lane = runout_lane

        t1.tc_lane._perform_infinite_runout()

        rig.afc.error.AFC_error.assert_called_once_with(
            f"Invalid Creator 5 infinite-spool replacement for extruder1: {runout_lane}")
        assert rig.scripts == []
        assert t1.tc_lane.status == AFCLaneState.TOOLED

    def test_unheated_source_keeps_print_paused(self, rig):
        t1, _ = self._runout(rig)
        t1.toolhead_extruder.heater.target_temp = 0.0

        t1.tc_lane._perform_infinite_runout()

        rig.afc.error.AFC_error.assert_called_once_with(
            "Creator 5 infinite-spool change failed; print remains paused: source hotend has "
            "no printable target temperature")
        rig.afc.error.pause_resume.send_pause_command.assert_called_once_with()
        rig.afc.error.pause_resume.send_resume_command.assert_not_called()
        assert rig.heaters.calls == []
        assert rig.scripts == []
        assert rig.afc.position_saved is True

    def test_failed_grab_keeps_print_paused_and_mapping(self, rig):
        t1, t2 = self._runout(rig)
        rig.gcode_move.base_position[:2] = [2.5, -1.25]
        rig.gcode_move.homing_position[:2] = [2.5, -1.25]
        rig.gcode_move.last_position[:2] = [102.5, 98.75]
        saved_position = list(rig.gcode_move.last_position)
        rig.board.select_fault = "T2 grab switch not confirmed"

        t1.tc_lane._perform_infinite_runout()

        rig.afc.error.AFC_error.assert_called_once_with(
            "Creator 5 infinite-spool change failed; print remains paused: T2 grab switch "
            "not confirmed")
        rig.afc.error.pause_resume.send_resume_command.assert_not_called()
        assert rig.scripts == ["C5_TOOL_SELECT T=2"]
        assert rig.gcode_move.moves == []
        assert rig.afc.in_toolchange is False
        assert rig.afc.position_saved is True
        assert rig.afc.last_gcode_position == pytest.approx(saved_position)
        assert t1.tc_lane.status == AFCLaneState.TOOLED
        assert t2.tc_lane.status == AFCLaneState.LOADED

    def test_spare_that_did_not_attach_keeps_print_paused(self, rig):
        t1, t2 = self._runout(rig)
        rig.board.attach_on_select = False

        t1.tc_lane._perform_infinite_runout()

        rig.afc.error.AFC_error.assert_called_once_with(
            "Creator 5 infinite-spool change failed; print remains paused: replacement tool "
            "did not attach")
        rig.afc.error.pause_resume.send_resume_command.assert_not_called()
        assert rig.scripts == ["C5_TOOL_SELECT T=2"]
        assert rig.gcode_move.moves == []
        assert t1.tc_lane.status == AFCLaneState.TOOLED
        assert t2.tc_lane.status == AFCLaneState.LOADED
