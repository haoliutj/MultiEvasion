"""
validate dos extension attack

"""

import copy, lief, math,operator,struct
import numpy as np
from attacks import attack_utils

data_path = '../data/putty.exe'
first_n_byte=102400
preferable_extension_amount = 512

with open(data_path,'rb') as f:
    tmp = [i for i in f.read()[:first_n_byte]]

X = copy.deepcopy(tmp)

f_byte = bytearray(list(np.array(X,dtype=int)))

X1 = list(f_byte)

eq = operator.eq(X,X1)
print(f'two byte lists is equal {eq}')

liefpe = lief.PE.parse(raw=f_byte)   # parse on list of integer
section_file_alignment = liefpe.optional_header.file_alignment
if section_file_alignment == 0:
    pass

pe_header_start_offset = liefpe.dos_header.addressof_new_exeheader
extension_amount = int(math.ceil(preferable_extension_amount/section_file_alignment))*section_file_alignment
index_to_perturb = list(range(2,0x3c)) + list(range(0x40,pe_header_start_offset+extension_amount))

"shift/alignment pe header offset to the amount"
# x = attack_utils.shift_pe_header_offset(liefpe,x,extension_amount)
f_byte_1 = copy.deepcopy(f_byte)
f_byte[0x3c:0x40] = struct.pack("<I", pe_header_start_offset + extension_amount) # update pe header offset

ttt = list(f_byte)
X2= ttt[59:64]
X3 = X1[59:64]
f2 = f_byte[0x3c:0x40]
f3 = f_byte_1[0x3c:0x40]

[f_byte.insert(pe_header_start_offset,0) for _ in range(extension_amount)]

"shift/alignmet sections' offset to the amount"
for i,_ in enumerate(liefpe.sections):
    # x = attack_utils.shift_pointer_offset_to_section_content(liefpe,bytearray(x),i,extension_amount,extension_amount)

    pe_header_start_index_shift_by = extension_amount
    amount = extension_amount
    entry_index = i
    pe_header_start_index = liefpe.dos_header.addressof_new_exeheader + pe_header_start_index_shift_by
    optional_header_size = liefpe.header.sizeof_optional_header
    coff_header_size = 24   ## common object file format: 20 bytes (the author used 24)
    section_entry_length = 40
    size_of_raw_data_pointer = 20
    shift_position = pe_header_start_index + optional_header_size + coff_header_size + \
                     (entry_index * section_entry_length) + size_of_raw_data_pointer

    # old_offset = struct.unpack("<I", x[shift_position : shift_position+4])[0]
    old_offset = struct.unpack("<I", f_byte[shift_position: shift_position + 4])[0]

    ## 16 进制
    old_offset_16 = struct.pack('I', old_offset)

    new_offset = old_offset + amount
    new_offset = struct.pack("<I", new_offset)

    f_byte[shift_position:shift_position + 4] = new_offset