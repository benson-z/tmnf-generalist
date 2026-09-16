"""Reversible, process-local TMInterface 2.2.1 camera-reset experiment.

Never writes DLL files. Checks the exact call instruction before replacing it.
Only use with the game process launched by render_probe.py.
"""
import ctypes as c
from ctypes import wintypes as w


class CameraResetPatch:
    def __init__(self, pid):
        self.k = c.WinDLL('kernel32', use_last_error=True)
        self.p = c.WinDLL('psapi', use_last_error=True)
        self.k.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        self.k.OpenProcess.restype = w.HANDLE
        self.k.CloseHandle.argtypes = [w.HANDLE]
        self.k.ReadProcessMemory.argtypes = [w.HANDLE, c.c_void_p, c.c_void_p, c.c_size_t, c.POINTER(c.c_size_t)]
        self.k.WriteProcessMemory.argtypes = self.k.ReadProcessMemory.argtypes
        self.k.VirtualProtectEx.argtypes = [w.HANDLE, c.c_void_p, c.c_size_t, w.DWORD, c.POINTER(w.DWORD)]
        self.k.FlushInstructionCache.argtypes = [w.HANDLE, c.c_void_p, c.c_size_t]
        self.p.EnumProcessModulesEx.argtypes = [w.HANDLE, c.POINTER(w.HMODULE), w.DWORD, c.POINTER(w.DWORD), w.DWORD]
        self.p.GetModuleBaseNameW.argtypes = [w.HANDLE, w.HMODULE, w.LPWSTR, w.DWORD]
        self.handle = self.k.OpenProcess(0x438, False, pid)
        if not self.handle:
            raise c.WinError(c.get_last_error())
        modules = (w.HMODULE * 1024)()
        needed = w.DWORD()
        self.check(self.p.EnumProcessModulesEx(self.handle, modules, c.sizeof(modules), c.byref(needed), 3))
        matches = []
        for module in modules[:needed.value // c.sizeof(w.HMODULE)]:
            name = c.create_unicode_buffer(260)
            if self.p.GetModuleBaseNameW(self.handle, module, name, 260) and name.value.lower() == 'tminterface.dll':
                matches.append(module)
        if len(matches) != 1:
            self.k.CloseHandle(self.handle)
            raise RuntimeError(f'Expected one TMInterface module, got {matches}')
        self.base = matches[0]
        self.address = self.base + 0x8437f
        self.original = b'\xff\x15' + (self.base + 0x2502e8).to_bytes(4, 'little')
        actual = self.read(self.address, 6)
        if actual != self.original:
            self.k.CloseHandle(self.handle)
            raise RuntimeError(f'Unsupported DLL: camera-reset call {actual.hex()} != {self.original.hex()}')
        self.enabled = False

    @staticmethod
    def check(ok):
        if not ok:
            raise c.WinError(c.get_last_error())

    def read(self, address, size):
        buf = c.create_string_buffer(size)
        done = c.c_size_t()
        self.check(self.k.ReadProcessMemory(self.handle, address, buf, size, c.byref(done)))
        if done.value != size:
            raise RuntimeError('Partial read')
        return buf.raw

    def set_enabled(self, enabled):
        if self.enabled == enabled:
            return
        expected = b'\x90' * 6 if self.enabled else self.original
        if self.read(self.address, 6) != expected:
            raise RuntimeError('Patch site changed unexpectedly')
        payload = b'\x90' * 6 if enabled else self.original
        old = w.DWORD()
        self.check(self.k.VirtualProtectEx(self.handle, self.address, 6, 0x40, c.byref(old)))
        try:
            done = c.c_size_t()
            self.check(self.k.WriteProcessMemory(self.handle, self.address, payload, 6, c.byref(done)))
            if done.value != 6:
                raise RuntimeError('Partial write')
            self.check(self.k.FlushInstructionCache(self.handle, self.address, 6))
        finally:
            ignored = w.DWORD()
            self.check(self.k.VirtualProtectEx(self.handle, self.address, 6, old.value, c.byref(ignored)))
        self.enabled = enabled

    def close(self):
        try:
            self.set_enabled(False)
        finally:
            self.k.CloseHandle(self.handle)
