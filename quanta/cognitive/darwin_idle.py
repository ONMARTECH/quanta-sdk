"""Darwin Mach kernel QoS, Apple Silicon hardware quiescence, and telemetry bindings.

Pillar 3: Autonomous Biomorphic Subconscious Mind-Wandering & Anticipatory Prospection Engine.
Implements low-level thread scheduling (QOS_CLASS_BACKGROUND = 0x09), I/O throttling,
Mach host CPU tick statistics, thermal pressure monitoring, and IOKit power telemetry.
Includes robust cross-platform fallbacks for Linux and Windows.
"""

from __future__ import annotations

import ctypes
import os
import platform
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# ============================================================================
# Darwin QoS Architecture Constants (<sys/qos.h>)
# ============================================================================
# Authoritative note: In Apple's sys/qos.h, QOS_CLASS_BACKGROUND is 0x09 (9),
# whereas 0x11 is QOS_CLASS_UTILITY. Subconscious execution requires 0x09 to
# guarantee scheduling strictly on Apple Silicon Efficiency cores (E-cores).
QOS_CLASS_USER_INTERACTIVE: int = 0x21
QOS_CLASS_USER_INITIATED: int = 0x19
QOS_CLASS_DEFAULT: int = 0x15
QOS_CLASS_UTILITY: int = 0x11
QOS_CLASS_BACKGROUND: int = 0x09
QOS_CLASS_UNSPECIFIED: int = 0x00
QOS_MIN_RELATIVE_PRIORITY: int = -15

# ============================================================================
# Darwin I/O Policy Constants (<sys/resource.h>)
# ============================================================================
IOPOL_TYPE_DISK: int = 0
IOPOL_SCOPE_PROCESS: int = 0
IOPOL_SCOPE_THREAD: int = 1

IOPOL_DEFAULT: int = 0
IOPOL_IMPORTANT: int = 1
IOPOL_PASSIVE: int = 2
IOPOL_THROTTLE: int = 3
IOPOL_UTILITY: int = 4
IOPOL_STANDARD: int = 5

# ============================================================================
# Mach Host Statistics Constants (<mach/host_info.h>, <mach/processor_info.h>)
# ============================================================================
HOST_CPU_LOAD_INFO: int = 3
PROCESSOR_CPU_LOAD_INFO: int = 2
CPU_STATE_USER: int = 0
CPU_STATE_SYSTEM: int = 1
CPU_STATE_IDLE: int = 2
CPU_STATE_NICE: int = 3
CPU_STATE_MAX: int = 4
HOST_CPU_LOAD_INFO_COUNT: int = 4

# ============================================================================
# Thermal Pressure State Constants (NSProcessInfoThermalState / OSThermalPressure)
# ============================================================================
THERMAL_STATE_NOMINAL: int = 0
THERMAL_STATE_FAIR: int = 1
THERMAL_STATE_SERIOUS: int = 2
THERMAL_STATE_CRITICAL: int = 3

IS_DARWIN: bool = platform.system() == "Darwin"
IS_LINUX: bool = platform.system() == "Linux"
IS_WINDOWS: bool = platform.system() == "Windows"


class HostCpuLoadInfo(ctypes.Structure):
    """Mach host CPU load info struct (4 32-bit unsigned integers)."""

    _fields_ = [("cpu_ticks", ctypes.c_uint * CPU_STATE_MAX)]


@dataclass(frozen=True)
class CpuTickSnapshot:
    """Snapshot of cumulative CPU ticks at a specific point in time."""

    user: int
    system: int
    idle: int
    nice: int
    timestamp: float

    @property
    def total(self) -> int:
        """Return aggregate ticks across all core states."""
        return self.user + self.system + self.idle + self.nice

    def __getitem__(self, idx: int) -> int:
        """Subscript indexing: [0]->user, [1]->system, [2]->idle, [3]->nice."""
        fields = (self.user, self.system, self.idle, self.nice)
        if 0 <= idx < 4:
            return fields[idx]
        raise IndexError(f"CpuTickSnapshot index out of range: {idx}")

    def __len__(self) -> int:
        """Return number of core state tick counters."""
        return 4

    def __iter__(self) -> Iterator[int]:
        """Yield (user, system, idle, nice) ticks for iteration or tuple unpacking."""
        return iter((self.user, self.system, self.idle, self.nice))


@dataclass(frozen=True)
class PowerTelemetry:
    """Power source and battery health telemetry."""

    is_ac_powered: bool
    battery_level_pct: int | None
    is_charging: bool
    time_to_empty_min: int | None
    battery_health: str


@dataclass(frozen=True)
class HardwareQuiescenceState:
    """Comprehensive hardware quiescence state evaluation."""

    is_quiescent: bool
    cpu_load_pct: float
    cpu_idle_pct: float
    thermal_pressure_level: int
    power_source: str
    battery_pct: int | None
    e_core_idle_pct: float | None = None
    p_core_idle_pct: float | None = None


class DarwinIdleMonitor:
    """Interfaces directly with Darwin Mach kernel, QoS, and IOKit subsystems.

    Provides high-performance, zero-dependency C-bindings via ctypes to monitor
    hardware quiescence, schedule background threads on Apple Silicon E-cores,
    and enforce thermal and thermodynamic power budget constraints.
    """

    def __init__(self) -> None:
        self.is_darwin: bool = IS_DARWIN
        self._lock: threading.Lock = threading.Lock()
        self._libsystem: ctypes.CDLL | None = None
        self._iokit: ctypes.CDLL | None = None
        self._cf: ctypes.CDLL | None = None
        self._thermal_token: ctypes.c_int = ctypes.c_int(0)
        self._last_snapshot: CpuTickSnapshot | None = None

        if self.is_darwin:
            self._init_darwin_bindings()

        # Seed initial CPU ticks snapshot
        try:
            self._last_snapshot = self.get_global_cpu_ticks()
        except Exception:
            self._last_snapshot = None

    def _init_darwin_bindings(self) -> None:
        """Initialize Mach, pthread, and IOKit C-level function pointers."""
        try:
            # 1. libSystem symbols are directly in process address space
            lib = ctypes.CDLL(None, use_errno=True)
            self._libsystem = lib

            # pthread QoS
            lib.pthread_set_qos_class_self_np.restype = ctypes.c_int
            lib.pthread_set_qos_class_self_np.argtypes = [ctypes.c_uint, ctypes.c_int]

            lib.pthread_get_qos_class_np.restype = ctypes.c_int
            lib.pthread_get_qos_class_np.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint),
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.pthread_self.restype = ctypes.c_void_p
            lib.pthread_self.argtypes = []
            lib.qos_class_self.restype = ctypes.c_uint
            lib.qos_class_self.argtypes = []

            # I/O Policy
            lib.setiopolicy_np.restype = ctypes.c_int
            lib.setiopolicy_np.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
            lib.getiopolicy_np.restype = ctypes.c_int
            lib.getiopolicy_np.argtypes = [ctypes.c_int, ctypes.c_int]

            # Mach host and task
            lib.mach_host_self.restype = ctypes.c_uint
            lib.mach_host_self.argtypes = []
            lib.mach_task_self.restype = ctypes.c_uint
            lib.mach_task_self.argtypes = []

            # Mach statistics
            if hasattr(lib, "host_statistics64"):
                lib.host_statistics64.restype = ctypes.c_int
                lib.host_statistics64.argtypes = [
                    ctypes.c_uint,
                    ctypes.c_int,
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint),
                ]
            if hasattr(lib, "host_statistics"):
                lib.host_statistics.restype = ctypes.c_int
                lib.host_statistics.argtypes = [
                    ctypes.c_uint,
                    ctypes.c_int,
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint),
                ]

            # Processor per-core statistics and VM deallocation
            if hasattr(lib, "host_processor_info"):
                lib.host_processor_info.restype = ctypes.c_int
                lib.host_processor_info.argtypes = [
                    ctypes.c_uint,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_uint),
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.POINTER(ctypes.c_uint),
                ]
            if hasattr(lib, "vm_deallocate"):
                lib.vm_deallocate.restype = ctypes.c_int
                lib.vm_deallocate.argtypes = [
                    ctypes.c_uint,
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                ]

            # notify(3) thermal pressure
            if hasattr(lib, "notify_register_check") and hasattr(lib, "notify_get_state"):
                lib.notify_register_check.restype = ctypes.c_int
                lib.notify_register_check.argtypes = [
                    ctypes.c_char_p,
                    ctypes.POINTER(ctypes.c_int),
                ]
                lib.notify_get_state.restype = ctypes.c_int
                lib.notify_get_state.argtypes = [
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_uint64),
                ]
                token = ctypes.c_int()
                lib.notify_register_check(
                    b"com.apple.system.thermalpressurelevel",
                    ctypes.byref(token),
                )
                self._thermal_token = token

            # Objective-C runtime for NSProcessInfo.thermalState
            if hasattr(lib, "objc_getClass") and hasattr(lib, "sel_registerName"):
                lib.objc_getClass.restype = ctypes.c_void_p
                lib.objc_getClass.argtypes = [ctypes.c_char_p]
                lib.sel_registerName.restype = ctypes.c_void_p
                lib.sel_registerName.argtypes = [ctypes.c_char_p]

            # 2. IOKit & CoreFoundation for battery/power telemetry
            try:
                self._iokit = ctypes.cdll.LoadLibrary(
                    "/System/Library/Frameworks/IOKit.framework/IOKit"
                )
                self._cf = ctypes.cdll.LoadLibrary(
                    "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
                )

                self._iokit.IOPSCopyPowerSourcesInfo.restype = ctypes.c_void_p
                self._iokit.IOPSCopyPowerSourcesInfo.argtypes = []
                self._iokit.IOPSCopyPowerSourcesList.restype = ctypes.c_void_p
                self._iokit.IOPSCopyPowerSourcesList.argtypes = [ctypes.c_void_p]
                self._iokit.IOPSGetPowerSourceDescription.restype = ctypes.c_void_p
                self._iokit.IOPSGetPowerSourceDescription.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                ]
                self._iokit.IOPSGetProvidingPowerSourceType.restype = ctypes.c_void_p
                self._iokit.IOPSGetProvidingPowerSourceType.argtypes = [ctypes.c_void_p]

                self._cf.CFArrayGetCount.restype = ctypes.c_long
                self._cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
                self._cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
                self._cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
                self._cf.CFDictionaryGetValue.restype = ctypes.c_void_p
                self._cf.CFDictionaryGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                self._cf.CFStringCreateWithCString.restype = ctypes.c_void_p
                self._cf.CFStringCreateWithCString.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_uint,
                ]
                self._cf.CFStringGetCString.restype = ctypes.c_bool
                self._cf.CFStringGetCString.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_long,
                    ctypes.c_uint,
                ]
                self._cf.CFNumberGetValue.restype = ctypes.c_bool
                self._cf.CFNumberGetValue.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_void_p,
                ]
                self._cf.CFBooleanGetValue.restype = ctypes.c_bool
                self._cf.CFBooleanGetValue.argtypes = [ctypes.c_void_p]
                self._cf.CFGetTypeID.restype = ctypes.c_ulong
                self._cf.CFGetTypeID.argtypes = [ctypes.c_void_p]
                self._cf.CFStringGetTypeID.restype = ctypes.c_ulong
                self._cf.CFStringGetTypeID.argtypes = []
                self._cf.CFNumberGetTypeID.restype = ctypes.c_ulong
                self._cf.CFNumberGetTypeID.argtypes = []
                self._cf.CFBooleanGetTypeID.restype = ctypes.c_ulong
                self._cf.CFBooleanGetTypeID.argtypes = []
                self._cf.CFRelease.restype = None
                self._cf.CFRelease.argtypes = [ctypes.c_void_p]
            except Exception:
                self._iokit = None
                self._cf = None
        except Exception:
            self._libsystem = None

    def set_background_qos(self) -> bool:
        """Enforces QOS_CLASS_BACKGROUND (0x09) and IOPOL_THROTTLE.

        On Darwin, assigns QOS_CLASS_BACKGROUND (0x09, relative priority 0)
        via pthread_set_qos_class_self_np and sets disk I/O policy to
        IOPOL_THROTTLE (3) via setiopolicy_np.
        On Linux/Windows, falls back gracefully to low process/thread priorities.

        Returns:
            bool: True if priority/QoS was successfully configured.
        """
        if self.is_darwin and self._libsystem is not None:
            ret_qos = 0
            ret_io = 0
            try:
                # 1. Set disk I/O throttle policy at process scope
                ret_io = self._libsystem.setiopolicy_np(
                    IOPOL_TYPE_DISK, IOPOL_SCOPE_PROCESS, IOPOL_THROTTLE
                )
            except Exception:
                ret_io = -1

            try:
                # 2. Set pthread QoS to BACKGROUND (0x09, priority 0)
                ret_qos = self._libsystem.pthread_set_qos_class_self_np(
                    QOS_CLASS_BACKGROUND, 0
                )
            except Exception:
                ret_qos = -1

            return (ret_qos == 0) and (ret_io == 0)

        # Linux Fallback: setpriority to 19 (lowest CPU priority)
        if IS_LINUX:
            try:
                os.setpriority(os.PRIO_PROCESS, 0, 19)
                return True
            except Exception:
                try:
                    os.nice(19)
                    return True
                except Exception:
                    return False

        # Windows Fallback: IDLE_PRIORITY_CLASS (0x40)
        if IS_WINDOWS:
            try:
                windll = getattr(ctypes, "windll", None)
                if windll is not None:
                    k32 = windll.kernel32
                    handle = k32.GetCurrentProcess()
                    res_proc = k32.SetPriorityClass(handle, 0x00000040)
                    thr_handle = k32.GetCurrentThread()
                    k32.SetThreadPriority(thr_handle, -15)  # THREAD_PRIORITY_IDLE
                    return bool(res_proc)
            except Exception:
                return False

        return False

    def get_active_qos(self) -> tuple[int, int]:
        """Query active pthread QoS class and relative priority.

        Returns:
            tuple[int, int]: (qos_class, relative_priority).
        """
        if not self.is_darwin or self._libsystem is None:
            return (0, 0)
        thread = self._libsystem.pthread_self()
        if not thread:
            return (0, 0)
        qos = ctypes.c_uint(0)
        prio = ctypes.c_int(0)
        err = self._libsystem.pthread_get_qos_class_np(
            thread, ctypes.byref(qos), ctypes.byref(prio)
        )
        if err != 0:
            return (0, 0)
        return (qos.value, prio.value)

    def get_global_cpu_ticks(self) -> CpuTickSnapshot:
        """Retrieve cumulative CPU ticks from Mach kernel or platform counters.

        Returns:
            CpuTickSnapshot: Snapshot with user, system, idle, nice ticks.
        """
        now = time.time()
        if self.is_darwin and self._libsystem is not None:
            info = HostCpuLoadInfo()
            count = ctypes.c_uint(HOST_CPU_LOAD_INFO_COUNT)
            host = self._libsystem.mach_host_self()
            ret = -1
            if hasattr(self._libsystem, "host_statistics64"):
                ret = self._libsystem.host_statistics64(
                    host, HOST_CPU_LOAD_INFO, ctypes.byref(info), ctypes.byref(count)
                )
            if ret != 0 and hasattr(self._libsystem, "host_statistics"):
                ret = self._libsystem.host_statistics(
                    host, HOST_CPU_LOAD_INFO, ctypes.byref(info), ctypes.byref(count)
                )

            if ret == 0:
                return CpuTickSnapshot(
                    user=int(info.cpu_ticks[CPU_STATE_USER]),
                    system=int(info.cpu_ticks[CPU_STATE_SYSTEM]),
                    idle=int(info.cpu_ticks[CPU_STATE_IDLE]),
                    nice=int(info.cpu_ticks[CPU_STATE_NICE]),
                    timestamp=now,
                )

        # Linux Fallback: parse /proc/stat
        if IS_LINUX:
            try:
                proc_stat = Path("/proc/stat")
                if proc_stat.exists():
                    with open(proc_stat, encoding="utf-8") as f:
                        for line in f:
                            if line.startswith("cpu "):
                                parts = [int(p) for p in line.split()[1:]]
                                u = parts[0]
                                n = parts[1] if len(parts) > 1 else 0
                                s = parts[2] if len(parts) > 2 else 0
                                i = parts[3] if len(parts) > 3 else 0
                                return CpuTickSnapshot(
                                    user=u,
                                    system=s,
                                    idle=i,
                                    nice=n,
                                    timestamp=now,
                                )
            except Exception:
                pass

        # Windows Fallback: GetSystemTimes
        if IS_WINDOWS:
            try:
                windll = getattr(ctypes, "windll", None)
                if windll is not None:

                    class FILETIME(ctypes.Structure):
                        _fields_ = [
                            ("dwLowDateTime", ctypes.c_uint),
                            ("dwHighDateTime", ctypes.c_uint),
                        ]

                    def to_int(ft: FILETIME) -> int:
                        return int((int(ft.dwHighDateTime) << 32) + int(ft.dwLowDateTime))

                    idle_time = FILETIME()
                    kernel_time = FILETIME()
                    user_time = FILETIME()
                    k32 = windll.kernel32
                    if k32.GetSystemTimes(
                        ctypes.byref(idle_time),
                        ctypes.byref(kernel_time),
                        ctypes.byref(user_time),
                    ):
                        i_val = to_int(idle_time)
                        k_val = to_int(kernel_time)
                        u_val = to_int(user_time)
                        sys_val = max(0, k_val - i_val)
                        return CpuTickSnapshot(
                            user=u_val,
                            system=sys_val,
                            idle=i_val,
                            nice=0,
                            timestamp=now,
                        )
            except Exception:
                pass

        # Default fallback: safe baseline
        return CpuTickSnapshot(user=0, system=0, idle=100, nice=0, timestamp=now)

    def get_per_core_cpu_ticks(self) -> list[CpuTickSnapshot]:
        """Retrieve per-core cumulative CPU ticks using host_processor_info.

        Uses Mach kernel host_processor_info with PROCESSOR_CPU_LOAD_INFO to
        extract CPU tick snapshots for all logical processor cores. Safely
        deallocates Mach VM memory via vm_deallocate.

        Returns:
            list[CpuTickSnapshot]: Cumulative tick snapshot for each logical processor.
        """
        now = time.time()
        if self.is_darwin and self._libsystem is not None:
            lib = self._libsystem
            if hasattr(lib, "host_processor_info") and hasattr(lib, "vm_deallocate"):
                try:
                    host = lib.mach_host_self()
                    proc_count = ctypes.c_uint(0)
                    proc_info_ptr = ctypes.c_void_p(0)
                    info_count = ctypes.c_uint(0)

                    ret = lib.host_processor_info(
                        host,
                        PROCESSOR_CPU_LOAD_INFO,
                        ctypes.byref(proc_count),
                        ctypes.byref(proc_info_ptr),
                        ctypes.byref(info_count),
                    )

                    if ret == 0 and proc_info_ptr.value:
                        snapshots: list[CpuTickSnapshot] = []
                        num_procs = proc_count.value
                        total_elements = info_count.value
                        try:
                            if total_elements >= num_procs * CPU_STATE_MAX:
                                int_array = ctypes.cast(
                                    proc_info_ptr,
                                    ctypes.POINTER(ctypes.c_uint * (num_procs * CPU_STATE_MAX)),
                                ).contents
                                for i in range(num_procs):
                                    offset = i * CPU_STATE_MAX
                                    snapshots.append(
                                        CpuTickSnapshot(
                                            user=int(int_array[offset + CPU_STATE_USER]),
                                            system=int(int_array[offset + CPU_STATE_SYSTEM]),
                                            idle=int(int_array[offset + CPU_STATE_IDLE]),
                                            nice=int(int_array[offset + CPU_STATE_NICE]),
                                            timestamp=now,
                                        )
                                    )
                            return snapshots
                        finally:
                            task = lib.mach_task_self()
                            size = info_count.value * ctypes.sizeof(ctypes.c_uint)
                            lib.vm_deallocate(
                                task,
                                ctypes.c_size_t(proc_info_ptr.value),
                                ctypes.c_size_t(size),
                            )
                except Exception:
                    pass

        # Linux Fallback: parse /proc/stat per-core lines (cpu0, cpu1, ...)
        if IS_LINUX:
            try:
                proc_stat = Path("/proc/stat")
                if proc_stat.exists():
                    linux_snapshots: list[CpuTickSnapshot] = []
                    with open(proc_stat, encoding="utf-8") as f:
                        for line in f:
                            if line.startswith("cpu") and not line.startswith("cpu "):
                                parts = [int(p) for p in line.split()[1:]]
                                u = parts[0]
                                n = parts[1] if len(parts) > 1 else 0
                                s = parts[2] if len(parts) > 2 else 0
                                i = parts[3] if len(parts) > 3 else 0
                                linux_snapshots.append(
                                    CpuTickSnapshot(
                                        user=u,
                                        system=s,
                                        idle=i,
                                        nice=n,
                                        timestamp=now,
                                    )
                                )
                    return linux_snapshots
            except Exception:
                pass

        return []

    def get_cpu_quiescence(self) -> float:
        """Calculate CPU idle ratio in [0.0, 1.0] from Mach tick deltas.

        Calculates tick differences (delta_user, delta_system, delta_idle, delta_nice)
        between consecutive calls using modular unsigned 32-bit arithmetic (& 0xFFFFFFFF)
        to prevent rollover anomalies. Synchronized across concurrent threads via self._lock.
        If insufficient time has elapsed to obtain non-zero deltas, performs a micro-sleep
        sample (20ms) to ensure true hardware measurement.

        Returns:
            float: Quiescence ratio in [0.0, 1.0], where 1.0 means 100% idle.
        """
        with self._lock:
            curr = self.get_global_cpu_ticks()
            prev = self._last_snapshot

            if prev is None:
                self._last_snapshot = curr
                time.sleep(0.02)
                curr = self.get_global_cpu_ticks()
                prev = self._last_snapshot

            du = (curr[0] - prev[0]) & 0xFFFFFFFF
            ds = (curr[1] - prev[1]) & 0xFFFFFFFF
            di = (curr[2] - prev[2]) & 0xFFFFFFFF
            dn = (curr[3] - prev[3]) & 0xFFFFFFFF
            d_total = du + ds + di + dn

            # If zero ticks elapsed (e.g. called back-to-back in microseconds), sample briefly
            if d_total <= 0:
                time.sleep(0.02)
                curr = self.get_global_cpu_ticks()
                du = (curr[0] - prev[0]) & 0xFFFFFFFF
                ds = (curr[1] - prev[1]) & 0xFFFFFFFF
                di = (curr[2] - prev[2]) & 0xFFFFFFFF
                dn = (curr[3] - prev[3]) & 0xFFFFFFFF
                d_total = du + ds + di + dn

            self._last_snapshot = curr

            if d_total <= 0:
                return 1.0

            idle_ratio = di / d_total
            return max(0.0, min(1.0, float(idle_ratio)))

    def compute_cpu_utilization(self) -> tuple[float, float]:
        """Return (load_percentage, idle_percentage) relative to last sample.

        Returns:
            tuple[float, float]: (load_pct, idle_pct) in range [0.0, 100.0].
        """
        quiescence = self.get_cpu_quiescence()
        idle_pct = quiescence * 100.0
        load_pct = max(0.0, 100.0 - idle_pct)
        return (round(load_pct, 2), round(idle_pct, 2))

    def _get_thermal_state_objc(self) -> int | None:
        """Query NSProcessInfo.thermalState directly via ObjC runtime."""
        if not self.is_darwin or self._libsystem is None:
            return None
        try:
            lib = self._libsystem
            cls = lib.objc_getClass(b"NSProcessInfo")
            sel_pi = lib.sel_registerName(b"processInfo")
            sel_ts = lib.sel_registerName(b"thermalState")
            if not (cls and sel_pi and sel_ts):
                return None
            msg_ptr = ctypes.CFUNCTYPE(
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
            )(("objc_msgSend", lib))
            msg_long = ctypes.CFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p
            )(("objc_msgSend", lib))

            proc_info = msg_ptr(cls, sel_pi)
            if not proc_info:
                return None
            state = msg_long(proc_info, sel_ts)
            if 0 <= state <= 3:
                return int(state)
        except Exception:
            pass
        return None

    def _get_thermal_state_notify(self) -> int | None:
        """Query notify(3) com.apple.system.thermalpressurelevel."""
        if not self.is_darwin or self._libsystem is None:
            return None
        try:
            lib = self._libsystem
            if self._thermal_token.value == 0:
                token = ctypes.c_int()
                if hasattr(lib, "notify_register_check"):
                    lib.notify_register_check(
                        b"com.apple.system.thermalpressurelevel",
                        ctypes.byref(token),
                    )
                    self._thermal_token = token
            if self._thermal_token.value != 0 and hasattr(lib, "notify_get_state"):
                raw_state = ctypes.c_uint64(0)
                err = lib.notify_get_state(
                    self._thermal_token.value, ctypes.byref(raw_state)
                )
                if err == 0:
                    val = raw_state.value
                    if val == 0:
                        return THERMAL_STATE_NOMINAL
                    elif val in (1, 10, 20):
                        return THERMAL_STATE_FAIR
                    elif val in (2, 30):
                        return THERMAL_STATE_SERIOUS
                    elif val >= 3:
                        return THERMAL_STATE_CRITICAL
        except Exception:
            pass
        return None

    def _get_thermal_state_linux(self) -> int | None:
        """Read Linux /sys/class/thermal/thermal_zone*/temp."""
        if not IS_LINUX:
            return None
        try:
            thermal_path = Path("/sys/class/thermal")
            if thermal_path.exists():
                max_temp_mc = 0
                for zone in thermal_path.glob("thermal_zone*/temp"):
                    try:
                        temp = int(zone.read_text().strip())
                        if temp > max_temp_mc:
                            max_temp_mc = temp
                    except Exception:
                        continue
                if max_temp_mc > 0:
                    temp_c = max_temp_mc / 1000.0
                    if temp_c >= 90.0:
                        return THERMAL_STATE_CRITICAL
                    elif temp_c >= 80.0:
                        return THERMAL_STATE_SERIOUS
                    elif temp_c >= 65.0:
                        return THERMAL_STATE_FAIR
                    return THERMAL_STATE_NOMINAL
        except Exception:
            pass
        return None

    def get_thermal_state(self) -> int:
        """Return Apple Silicon thermal state (0=Nominal, 1=Fair, 2=Serious, 3=Critical).

        Returns:
            int: 0=Nominal, 1=Fair, 2=Serious, 3=Critical.
        """
        if self.is_darwin:
            objc_state = self._get_thermal_state_objc()
            if objc_state is not None:
                return objc_state
            notify_state = self._get_thermal_state_notify()
            if notify_state is not None:
                return notify_state
        if IS_LINUX:
            linux_state = self._get_thermal_state_linux()
            if linux_state is not None:
                return linux_state
        return THERMAL_STATE_NOMINAL

    def get_thermal_pressure(self) -> int:
        """Alias for get_thermal_state returning integer level 0..3."""
        return self.get_thermal_state()

    def _get_fallback_power_telemetry(self) -> PowerTelemetry:
        """Cross-platform fallback for battery telemetry."""
        # Try psutil if available
        try:
            import psutil

            batt = psutil.sensors_battery()
            if batt is not None:
                secs = batt.secsleft if (batt.secsleft and batt.secsleft > 0) else None
                tte = int(secs / 60) if secs is not None else None
                return PowerTelemetry(
                    is_ac_powered=bool(batt.power_plugged),
                    battery_level_pct=int(batt.percent),
                    is_charging=bool(batt.power_plugged and batt.percent < 100),
                    time_to_empty_min=tte,
                    battery_health="Normal",
                )
        except Exception:
            pass

        # Linux /sys/class/power_supply
        if IS_LINUX:
            try:
                ps_path = Path("/sys/class/power_supply")
                if ps_path.exists():
                    is_ac = True
                    for ac in ps_path.glob("AC*/online"):
                        if ac.read_text().strip() == "0":
                            is_ac = False
                    cap: int | None = None
                    for bat in ps_path.glob("BAT*/capacity"):
                        cap = int(bat.read_text().strip())
                        break
                    return PowerTelemetry(
                        is_ac_powered=is_ac,
                        battery_level_pct=cap,
                        is_charging=not is_ac,
                        time_to_empty_min=None,
                        battery_health="Normal",
                    )
            except Exception:
                pass

        # Default fallback (e.g. desktop workstation)
        return PowerTelemetry(
            is_ac_powered=True,
            battery_level_pct=None,
            is_charging=False,
            time_to_empty_min=None,
            battery_health="Unknown",
        )

    def get_power_telemetry(self) -> PowerTelemetry:
        """Query battery and AC power status via IOKit.

        Returns:
            PowerTelemetry: Dataclass containing AC status, battery percentage,
            charging state, time-to-empty, and battery health.
        """
        if not self.is_darwin or self._iokit is None or self._cf is None:
            return self._get_fallback_power_telemetry()

        iokit = self._iokit
        cf = self._cf
        blob = None
        ps_list = None
        try:
            blob = iokit.IOPSCopyPowerSourcesInfo()
            if not blob:
                return self._get_fallback_power_telemetry()

            # Providing power source type (e.g., "AC Power", "Battery Power", "UPS Power")
            is_ac = True
            src_type_ref = iokit.IOPSGetProvidingPowerSourceType(blob)
            if src_type_ref:
                buf = ctypes.create_string_buffer(256)
                if cf.CFStringGetCString(src_type_ref, buf, 256, 0x08000100):
                    src_type = buf.value.decode("utf-8", errors="replace")
                    is_ac = src_type == "AC Power"

            ps_list = iokit.IOPSCopyPowerSourcesList(blob)
            if not ps_list:
                return PowerTelemetry(
                    is_ac_powered=is_ac,
                    battery_level_pct=None,
                    is_charging=False,
                    time_to_empty_min=None,
                    battery_health="Unavailable",
                )

            count = cf.CFArrayGetCount(ps_list)
            if count == 0:
                return PowerTelemetry(
                    is_ac_powered=is_ac,
                    battery_level_pct=None,
                    is_charging=False,
                    time_to_empty_min=None,
                    battery_health="None",
                )

            ps = cf.CFArrayGetValueAtIndex(ps_list, 0)
            desc = iokit.IOPSGetPowerSourceDescription(blob, ps)
            if not desc:
                return PowerTelemetry(
                    is_ac_powered=is_ac,
                    battery_level_pct=None,
                    is_charging=False,
                    time_to_empty_min=None,
                    battery_health="Unavailable",
                )

            str_tid = cf.CFStringGetTypeID()
            num_tid = cf.CFNumberGetTypeID()
            bool_tid = cf.CFBooleanGetTypeID()
            utf8 = 0x08000100

            def unpack(key_name: str) -> str | int | bool | None:
                cf_k = cf.CFStringCreateWithCString(
                    None, key_name.encode("utf-8"), utf8
                )
                if not cf_k:
                    return None
                try:
                    v_ref = cf.CFDictionaryGetValue(desc, cf_k)
                    if not v_ref:
                        return None
                    tid = cf.CFGetTypeID(v_ref)
                    if tid == str_tid:
                        buf = ctypes.create_string_buffer(256)
                        if cf.CFStringGetCString(v_ref, buf, 256, utf8):
                            return buf.value.decode("utf-8", errors="replace")
                    elif tid == num_tid:
                        val = ctypes.c_int64()
                        if cf.CFNumberGetValue(v_ref, 4, ctypes.byref(val)):
                            return val.value
                    elif tid == bool_tid:
                        return bool(cf.CFBooleanGetValue(v_ref))
                finally:
                    cf.CFRelease(cf_k)
                return None

            state_str = unpack("Power Source State")
            if isinstance(state_str, str):
                is_ac = state_str == "AC Power"
            cap = unpack("Current Capacity")
            charging = unpack("Is Charging")
            tte = unpack("Time to Empty")
            health = unpack("BatteryHealth")

            return PowerTelemetry(
                is_ac_powered=is_ac,
                battery_level_pct=int(cap) if isinstance(cap, int) else None,
                is_charging=bool(charging) if isinstance(charging, bool) else False,
                time_to_empty_min=int(tte) if (isinstance(tte, int) and tte > 0) else None,
                battery_health=str(health) if health is not None else "Normal",
            )
        except Exception:
            return self._get_fallback_power_telemetry()
        finally:
            if ps_list:
                cf.CFRelease(ps_list)
            if blob:
                cf.CFRelease(blob)

    def is_on_battery(self) -> bool:
        """Return True if system is running on battery power.

        Returns:
            bool: True if machine is running on battery power; False if AC or desktop.
        """
        telemetry = self.get_power_telemetry()
        return not telemetry.is_ac_powered

    def assess_quiescence(
        self, min_idle_pct: float = 70.0, max_thermal: int = 1
    ) -> HardwareQuiescenceState:
        """Comprehensive hardware quiescence state evaluation.

        Args:
            min_idle_pct: Minimum CPU idle percent threshold (0-100).
            max_thermal: Maximum allowable thermal pressure level (0-3).

        Returns:
            HardwareQuiescenceState: Current quiescence metrics.
        """
        quiescence = self.get_cpu_quiescence()
        idle_pct = round(quiescence * 100.0, 2)
        load_pct = round(max(0.0, 100.0 - idle_pct), 2)
        thermal = self.get_thermal_state()
        pwr = self.get_power_telemetry()

        is_quiescent = (idle_pct >= min_idle_pct) and (thermal <= max_thermal)

        return HardwareQuiescenceState(
            is_quiescent=is_quiescent,
            cpu_load_pct=load_pct,
            cpu_idle_pct=idle_pct,
            thermal_pressure_level=thermal,
            power_source="AC Power" if pwr.is_ac_powered else "Battery Power",
            battery_pct=pwr.battery_level_pct,
        )

    def is_system_idle(self, idle_threshold: float = 0.70, max_thermal: int = 1) -> bool:
        """Check if host is quiet enough for subconscious mind-wandering.

        Args:
            idle_threshold: Minimum required CPU idle ratio in [0.0, 1.0]. Default 0.70.
            max_thermal: Maximum permissible thermal state (default 1 = Fair).

        Returns:
            bool: True if host CPU idle ratio >= idle_threshold and thermal <= max_thermal.
        """
        quiescence = self.get_cpu_quiescence()
        thermal = self.get_thermal_state()
        return (quiescence >= idle_threshold) and (thermal <= max_thermal)


# ============================================================================
# Process-Wide Singleton & Public Module-Level API Functions
# ============================================================================

_default_monitor: DarwinIdleMonitor | None = None
_monitor_lock: threading.Lock = threading.Lock()


def get_default_monitor() -> DarwinIdleMonitor:
    """Return the process-wide default DarwinIdleMonitor singleton.

    Returns:
        DarwinIdleMonitor: Shared monitor instance.
    """
    global _default_monitor
    if _default_monitor is None:
        with _monitor_lock:
            if _default_monitor is None:
                _default_monitor = DarwinIdleMonitor()
    return _default_monitor


def get_per_core_cpu_ticks() -> list[CpuTickSnapshot]:
    """Retrieve per-core cumulative CPU ticks for all logical processors.

    Returns:
        list[CpuTickSnapshot]: List of tick snapshots, one per core.
    """
    return get_default_monitor().get_per_core_cpu_ticks()


def set_background_qos() -> bool:
    """Enforce QOS_CLASS_BACKGROUND (0x09) and IOPOL_THROTTLE.

    Returns:
        bool: True if background QoS and throttle policy were successfully set.
    """
    return get_default_monitor().set_background_qos()


def get_cpu_quiescence() -> float:
    """Return CPU idle ratio in [0.0, 1.0] from host_statistics64.

    Returns:
        float: Idle ratio in [0.0, 1.0].
    """
    return get_default_monitor().get_cpu_quiescence()


def get_thermal_state() -> int:
    """Return Apple Silicon thermal state (0=Nominal, 1=Fair, 2=Serious, 3=Critical).

    Returns:
        int: Thermal pressure level.
    """
    return get_default_monitor().get_thermal_state()


def is_on_battery() -> bool:
    """Return True if system is running on battery power.

    Returns:
        bool: True if on battery power.
    """
    return get_default_monitor().is_on_battery()


def is_system_idle(idle_threshold: float = 0.70, max_thermal: int = 1) -> bool:
    """Check if host is quiet enough for subconscious mind-wandering.

    Args:
        idle_threshold: Minimum required CPU idle ratio in [0.0, 1.0].
        max_thermal: Maximum permissible thermal state (0..3).

    Returns:
        bool: True if host is sufficiently quiet.
    """
    return get_default_monitor().is_system_idle(
        idle_threshold=idle_threshold, max_thermal=max_thermal
    )


__all__ = [
    "QOS_CLASS_USER_INTERACTIVE",
    "QOS_CLASS_USER_INITIATED",
    "QOS_CLASS_DEFAULT",
    "QOS_CLASS_UTILITY",
    "QOS_CLASS_BACKGROUND",
    "QOS_CLASS_UNSPECIFIED",
    "QOS_MIN_RELATIVE_PRIORITY",
    "IOPOL_TYPE_DISK",
    "IOPOL_SCOPE_PROCESS",
    "IOPOL_SCOPE_THREAD",
    "IOPOL_DEFAULT",
    "IOPOL_IMPORTANT",
    "IOPOL_PASSIVE",
    "IOPOL_THROTTLE",
    "IOPOL_UTILITY",
    "IOPOL_STANDARD",
    "HOST_CPU_LOAD_INFO",
    "PROCESSOR_CPU_LOAD_INFO",
    "CPU_STATE_USER",
    "CPU_STATE_SYSTEM",
    "CPU_STATE_IDLE",
    "CPU_STATE_NICE",
    "CPU_STATE_MAX",
    "HOST_CPU_LOAD_INFO_COUNT",
    "THERMAL_STATE_NOMINAL",
    "THERMAL_STATE_FAIR",
    "THERMAL_STATE_SERIOUS",
    "THERMAL_STATE_CRITICAL",
    "CpuTickSnapshot",
    "PowerTelemetry",
    "HardwareQuiescenceState",
    "DarwinIdleMonitor",
    "get_default_monitor",
    "get_per_core_cpu_ticks",
    "set_background_qos",
    "get_cpu_quiescence",
    "get_thermal_state",
    "is_on_battery",
    "is_system_idle",
]
