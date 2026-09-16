"""Read-only disassembly of TMInterface's render API registration/function."""
import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'out/render-tools'))
import pefile
from capstone import Cs, CS_ARCH_X86, CS_MODE_32

p = argparse.ArgumentParser()
p.add_argument('--address', type=lambda s: int(s, 0))
p.add_argument('--size', type=int, default=240)
p.add_argument('--game', action='store_true')
args = p.parse_args()
path = Path.home() / 'AppData/Local/TMLoader/database/TmForever/products/TMInterface/2.2.1/TMInterface.dll'
if args.game:
    path = Path.home() / 'AppData/Local/TMLoader/database/TmForever/products/TmForever/2.12.0/TmForever.exe'
pe = pefile.PE(str(path))
base = pe.OPTIONAL_HEADER.ImageBase
data = path.read_bytes()
dis = Cs(CS_ARCH_X86, CS_MODE_32)
def show(address, size):
    for ins in dis.disasm(pe.get_data(address - base, size), address):
        print(f'{ins.address:08x}: {ins.mnemonic:8s} {ins.op_str}')
if args.address:
    show(args.address, args.size)
else:
    print('Image base', hex(base))
    for name in (b'ForceGameRender', b'ResetCamera'):
        at = data.find(name)
        while at >= 0:
            start = data.rfind(b'\0', 0, at) + 1
            va = base + pe.get_rva_from_offset(start)
            print(name, hex(va), repr(data[start:data.find(b'\0', at)]))
            for pointer in (va, base + pe.get_rva_from_offset(at)):
                ref = data.find(struct.pack('<I', pointer))
                while ref >= 0:
                    address = base + pe.get_rva_from_offset(ref)
                    print('reference', hex(address))
                    show(address - 40, 110)
                    ref = data.find(struct.pack('<I', pointer), ref + 4)
            at = data.find(name, at + 1)
