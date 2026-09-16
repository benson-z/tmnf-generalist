"""Read selected routines from the running probe instance, without patching."""
import sys
sys.path.insert(0, 'out/render-tools')
from capstone import Cs, CS_ARCH_X86, CS_MODE_32
from tmnf_collect.launcher import _processes
from render_patch import CameraResetPatch

targets = [p for p in _processes() if '/tmnfml_id=19' in (p.get('CommandLine') or '')]
if len(targets) != 1:
    raise RuntimeError('Expected exactly one running probe instance')
p = CameraResetPatch(targets[0]['ProcessId'])
try:
    d = Cs(CS_ARCH_X86, CS_MODE_32)
    for rva in (0x250334, 0x2502e8):
        address = int.from_bytes(p.read(p.base + rva, 4), 'little')
        print('Pointer RVA', hex(rva), 'target', hex(address))
        for i in d.disasm(p.read(address, 256), address):
            print(hex(i.address), i.mnemonic, i.op_str)
            if i.mnemonic == 'ret':
                break
finally:
    p.close()
