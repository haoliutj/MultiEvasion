import lief
import numpy as np
import magic
import struct
import math

path = '../../data/all_file/benign_8'
x_init = lief.PE.parse(path)
# print(x_init)
print(f'size of optional header {x_init.header.sizeof_optional_header}')

ttt = x_init.dos_header.addressof_new_exeheader
print(ttt)



"get the extension index list of DOS"
preferable_extension_amount = 0x200
first_content_offset = x_init.sections[0].offset

section_file_alignment = x_init.optional_header.file_alignment
# the extending size will be 原section_file_alignment的整数倍, math.ceil(): round the number to its nearest number"
extension_amount = int(math.ceil(preferable_extension_amount / section_file_alignment)) * section_file_alignment
index_to_perturb = list(range(first_content_offset, first_content_offset + extension_amount))


# to see if file is PE file
# pe = magic.from_file(path)
# print(pe)

with open(path,'rb') as file:
    x_bytes = file.read()

print(f"pe bytes from 0x3c to 0x40 {x_bytes[0x3c:0x40]}")

pe_header_index = x_bytes.find(b'PE')
print(f'PE header starting index is {pe_header_index}')


int_x = np.frombuffer(x_bytes, dtype=np.uint8)

# must refer list of integer to parameter "raw" if parsed on list of integer
x_pe = lief.parse(raw=int_x) # to see if lief can parse on list of integer
hh = lief.parse(raw=bytearray(int_x))
print(f"lief parse on path, section file alignment {x_init.optional_header.file_alignment},pe starting index {x_init.dos_header.addressof_new_exeheader}, sections {x_init.sections[0].offset}")
print(f"lief parse on list of integer, section file alignment {x_pe.optional_header.file_alignment},pe starting index {x_pe.dos_header.addressof_new_exeheader},sections {x_pe.sections[0].offset}")

#
print("*"*50)
rrr = int_x
[np.insert(rrr,x_pe.dos_header.addressof_new_exeheader,0) for _ in range(5)]
ff = lief.parse(raw=bytearray(rrr))
print(f"After inserted, first section offset {ff.sections[0].offset}")
print("*"*50)



pe_position = int_x[0x3C:0x40]
# pe_position_shift = np.array([p-1 for p in pe_position])
pe_position_unpack:tuple = struct.unpack("<I", bytes(pe_position.astype(np.uint8))) # tuple

# reverse unpack
ori_bb =x_bytes[0x3c:0x40]
bb = struct.pack("I",pe_position_unpack[0])

pe_position_unpack = pe_position_unpack[0]
pe_header = x_bytes[pe_position_unpack:pe_position_unpack+2]
print(f'pe header {pe_header}, pe starting index {pe_position_unpack}')

indexes_to_perturb = [i for i in range(2,0x3c)] + [i for i in range(0x40,pe_position_unpack)]

print(f'length of bytes can be perturbed is {len(indexes_to_perturb)}')


