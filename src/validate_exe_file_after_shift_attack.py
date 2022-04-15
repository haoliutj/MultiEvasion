"""
validate dos extension attack

"""

import copy, lief, math,operator,struct
import numpy as np
from attacks import attack_utils

data_path = '../data/PEview.exe'
first_n_byte=102400
preferable_shift_amount = 512

with open(data_path,'rb') as f:
    tmp = [i for i in f.read()[:first_n_byte]]

X = copy.deepcopy(tmp)

f_byte = bytearray(list(np.array(X,dtype=int)))

X1 = list(f_byte)

eq = operator.eq(X,X1)
print(f'two byte lists is equal {eq}')


## return x and empty list if preferable_shift_amount is 0 or None
if not preferable_shift_amount:
    preferable_shift_amount = 512

liefpe = lief.PE.parse(raw=f_byte)
section_file_alignment = liefpe.optional_header.file_alignment
if section_file_alignment == 0:
    print('stop')
    pass

first_content_offset = liefpe.sections[0].offset
extension_amount = int(math.ceil(preferable_shift_amount/section_file_alignment)) * section_file_alignment
index_to_perturb = list(range(first_content_offset, first_content_offset+extension_amount))



"shift/alignmet sections' offset to the amount"
for i,_ in enumerate(liefpe.sections):
    # x = attack_utils.shift_pointer_offset_to_section_content(liefpe,bytearray(x),i,extension_amount,extension_amount)

    pe_header_start_index_shift_by = 0
    amount = extension_amount
    entry_index = i
    pe_header_start_index = liefpe.dos_header.addressof_new_exeheader + pe_header_start_index_shift_by
    optional_header_size = liefpe.header.sizeof_optional_header
    coff_header_size = 24       ## common object file format: 20 bytes (the author used 24)
    section_entry_length = 40
    size_of_raw_data_pointer = 20
    shift_position = pe_header_start_index + optional_header_size + coff_header_size + \
                     (entry_index * section_entry_length) + size_of_raw_data_pointer

    # old_offset = struct.unpack("<I", x[shift_position : shift_position+4])[0]
    old_offset = struct.unpack("<I", f_byte[shift_position: shift_position + 4])[0]

    ## 16 进制
    old_offset_16_ori = f_byte[shift_position: shift_position + 4]
    old_offset_16 = struct.pack('I', old_offset)

    new_offset = old_offset + amount
    new_offset = struct.pack("<I", new_offset)

    f_byte[shift_position:shift_position + 4] = new_offset

## update x: fill will 0s in the new created space
t = len(f_byte)
f_byte = f_byte[:first_content_offset] + b'\x00'*extension_amount + f_byte[first_content_offset:]
tt = len(f_byte)


