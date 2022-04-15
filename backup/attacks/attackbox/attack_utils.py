"""
subfunctions for attacks
based on works from https://github.com/zangobot/secml_malware

Including Attacks:
1. DOS Header Attack: all bytes except two magic numbers "MZ" [0x00,0x02]  and 4 bytes values at [0x3c,0x40) can not be changed
2. Extended DOS Header Attack: bytes from DOS Header Attacks, plus new extended space of DOS Header
3. Content Shift Attack: new created space between PE header and first section
4. Slack Attack: the unused space between sections (not include yet)

"""
import lief, struct, math
import numpy as np


def shift_pe_header_offset(liefpe,x:list,amount:int):
    """
    change the pe header offset to the modified value;
    initiated new extended space with 0;

    x: list of integer of binary file, (original raw bytes of binary file))
    liefpe: parsed result based on original raw bytes of binary file (list of integer)
    amount: number of bytes to shift/align

    return: x after pe header start index shifted/alignment
    """
    pe_header_start_offset = liefpe.dos_header.addressof_new_exeheader
    x[0x3c:0x40] = struct.pack("<I", pe_header_start_offset + amount) # update pe header offset

    ## insert 0 at position of pe_header_start_offset, repeat amount times
    [x.insert(pe_header_start_offset,0) for _ in range(amount)] # fill with 0s in new created space

    return x


def shift_pointer_offset_to_section_content(liefpe,x:bytearray,entry_index:int,amount:int,pe_header_start_index_shift_by: int=0):
    """
    shift/align the sections' offset to the modified value

    x: bytearray of binary file, [b'\0x00\0x03']
    liefpe: parsed result based on original raw bytes of binary file (list of integer)
    entry_index: index of each section
    amount: number of bytes to shift/align
    pe_header_start_index_shift_by: the value to shift pe header index

    return: x after sections' offset shifted/aligned
    """
    pe_header_start_index = liefpe.dos_header.addressof_new_exeheader + pe_header_start_index_shift_by
    optional_header_size = liefpe.header.sizeof_optional_header
    coff_header_size = 24
    section_entry_length = 40
    size_of_raw_data_pointer = 20
    shift_position = pe_header_start_index + optional_header_size + coff_header_size + \
                     (entry_index*section_entry_length) + size_of_raw_data_pointer

    # old_offset = struct.unpack("<I", x[shift_position : shift_position+4])[0]
    old_offset = struct.unpack("<I", x[shift_position : shift_position + 4])[0]
    new_offset = old_offset + amount
    new_offset = struct.pack("<I",new_offset)
    x[shift_position:shift_position+4] = new_offset

    return x


def shift_pe_header_by(x:list, preferable_extension_amount:int):
    """
    ---------> DOS Extension/Full DOS attack: <-------------
    magic numbers in DOS header "MZ"[0x00-0x02] and [0x3c-0x40] can't be modified;
    It's full DOS Header attack if preferable_extension_amount=0

    extend DOS header to preferable_extension_amount, initiated all extended space with 0
        - extend the times of section_alignment_offet,
        i.e.,int(math.ceil(preferable_extension_amount/section_file_alignment))*section_file_alignment
    shift/alignment pe header offset to the amount,
    shift/alignment section offset to the amount,

    x: list of integer of binary file
    preferable_extension_amount: integer, number of bytes to extend.

    return: shifted/aligned x and index to perturb
    """

    liefpe = lief.PE.parse(raw=x)   # parse on list of integer
    section_file_alignment = liefpe.optional_header.file_alignment
    if section_file_alignment == 0:
        return x, []

    pe_header_start_offset = liefpe.dos_header.addressof_new_exeheader

    ## if preferable_extension_amount < section_file_alignment
    ## --> math.ceil() still get result 1
    extension_amount = int(math.ceil(preferable_extension_amount / section_file_alignment)) * section_file_alignment
    index_to_perturb = list(range(2,0x3c)) + list(range(0x40,pe_header_start_offset+extension_amount))

    "shift/alignment pe header offset to the amount"
    x = shift_pe_header_offset(liefpe,x,extension_amount)

    "shift/alignmet sections' offset to the amount"
    for i,_ in enumerate(liefpe.sections):
        x = shift_pointer_offset_to_section_content(liefpe,bytearray(x),i,extension_amount,extension_amount)

    return x, index_to_perturb


def shift_section_by(x:list, preferable_shift_amount:int, pe_header_start_index_shift_by:int=0):
    """
    ------------> Content Shift Attack: <------------
    create space between PE header and the first section for perturbation;
    new created space initiated with 0;

    x: bytearray of integer of binary file
    preferable_extension_amount: integer, number of bytes to shift content (or to make space between PE header and first section)
    pe_header_start_index_shift_by: the value to shift pe header index. (need to provide if DOS Extension attack (i.e.,shift_pe_header_by) performed)

    return: x after shifted/alignment of Content Shift Attack; index to perturbation
    """
    ## return x and empty list if preferable_shift_amount is 0 or None
    if not preferable_shift_amount:
        return x, []

    liefpe = lief.PE.parse(raw=x)
    section_file_alignment = liefpe.optional_header.file_alignment
    if section_file_alignment == 0:
        return x, []

    first_content_offset = liefpe.sections[0].offset
    extension_amount = int(math.ceil(preferable_shift_amount/section_file_alignment)) * section_file_alignment
    index_to_perturb = list(range(first_content_offset, first_content_offset+extension_amount))

    for i,_ in enumerate(liefpe.sections):
        x = shift_pointer_offset_to_section_content(liefpe,x,i,extension_amount,pe_header_start_index_shift_by)

    ## update x: fill will 0s in the new created space
    x = x[:first_content_offset] + b'\x00'*extension_amount + x[first_content_offset:]

    return x, index_to_perturb


def partial_dos_attack(x:list):
    """
    partial dos attack, and no need to initiation here for partial dos attack as no new space created

    x: list of integer of binary file
    :return: orginal input x (no change), list of index that can be use to perturb in dos
    """
    index_to_perturb = list(range(2, 0x3c))

    return x,index_to_perturb


def slack_attack(x:bytearray=None, max_input_size:int=102400):
    """

    :param x: bytearray of input
    :param max_input_size: input size of classifier
    :return: sample after slack attack (list), position index of slack attack
    """
    try:
        liefpe = lief.PE.parse(x)
    except:
        return []
    index_to_perturb = []
    for s in liefpe.sections:
        if s.size > s.virtual_size:
            index_to_perturb.extend(list(range(min(max_input_size, s.offset+s.virtual_size),
                                               min(max_input_size, s.offset+s.size))))

    return x, index_to_perturb



def get_perturb_index_and_init_x_old_version(input=None,
                                             preferable_extension_amount:int=0,
                                             preferable_shift_amount:int=0x200,
                                             max_input_size:int=102400):
    """
    get the perturbation index, and initiated x with 0 if new space created by DOS_Extension and Content_Shift
    Full DOS attack: preferable_extension_amount = 0
    DOS Extension attack: preferable_extension_amount > 0
    content shift attack: preferable_shift_amount > 0
    Note: first Content_Shift attack, then DOS_extension attack, to avoid double update start index of pe header

    input: list of raw bytes of pe file
    preferable_extension_amount: number of bytes to extend in DOS_Extension attack. Default as 0 --> full DOS Attack
    preferable_shift_amount:
        - number of bytes to shift first content (i.e., number of space create between PE header and first section)
        - Default as 0x200 (512 bytes)
    first_n_bytes: -->int: the first n bytes (fixed length) feed into model

    return: initiated_x:list and index_to_perturb:list
    """
    x_init, index_to_perturb_section = shift_section_by(input, preferable_shift_amount=preferable_shift_amount,pe_header_start_index_shift_by=0)
    x_init, index_to_perturb_dos = shift_pe_header_by(x_init,preferable_extension_amount=preferable_extension_amount)

    """
    for my understanding: 
    the index_to_perturb in content shift attack should shift preferable_extension_amount,
    since the DOS extension attack has an follow-up impact on content shift attack,
    and first_content_offet does not change/update after DOS_Extension attack
    
    However, the author said the shifting offset should be the length of index_to_perturb_dos
    
    --> authors explained this on https://github.com/pralab/secml_malware/issues/11 
    """
    ## my understanding
    # index_to_perturb = index_to_perturb_dos + [i + preferable_extension_amount for i in index_to_perturb_section]

    ## shift attack's author's idea
    index_to_perturb = index_to_perturb_dos + [i + len(index_to_perturb_dos) for i in index_to_perturb_section]

    ## remove the perturb index that larger than the input length of detector (fist_n_bytes)
    index_to_perturb = np.array(index_to_perturb)
    index_to_perturb = index_to_perturb[index_to_perturb < max_input_size]
    index_to_perturb = list(index_to_perturb)


    return index_to_perturb, x_init


def get_perturb_index_and_init_x_v1(input=None,
                                 preferable_extension_amount: int = 0,
                                 preferable_shift_amount: int = 0x200,
                                 max_input_size: int = 102400,
                                 partial_dos: bool=False,
                                 content_shfit: bool=False):
    """
    get the perturbation index, and initiated x with 0 if new space created by DOS_Extension and Content_Shift
    1) Partial DOS attack: partial_dos = True
    2) Full DOS attack: preferable_extension_amount = 0 and preferable_shift_amount==0
    3) DOS Extension attack: preferable_extension_amount > 0 and preferable_shift_amount ==0
    4) content shift attack: content_shit=True and preferable_shift_amount > 0
    Note: first Content_Shift attack, then DOS_extension attack, to avoid double update start index of pe header

    input: list of raw bytes of pe file
    preferable_extension_amount: number of bytes to extend in DOS_Extension attack. Default as 0 --> full DOS Attack
    preferable_shift_amount:
        - number of bytes to shift first content (i.e., number of space create between PE header and first section)
        - Default as 0x200 (512 bytes)
    max_input_size: -->int: the first n bytes (fixed length) feed into model
    partial_dos: -->bool: true--> only perform partial dos attack
    content_shift: bool: true --> only perform content shift attack

    return: initiated_x:list and index_to_perturb:list
    """
    if partial_dos=='True' and content_shfit== 'False':
        ## partial DOS attack
        x_init, index_to_perturb = partial_dos_attack(input)
    elif content_shfit== 'True' and partial_dos== 'False':
        ## content shit attack
        x_init, index_to_perturb = shift_section_by(input,
                                                    preferable_shift_amount=preferable_shift_amount,
                                                    pe_header_start_index_shift_by=0)
    elif content_shfit == 'True' and partial_dos== 'True':
        ## content shift attack + partial dos
        x_init, index_to_perturb_section = shift_section_by(input,
                                                            preferable_shift_amount=preferable_shift_amount,
                                                            pe_header_start_index_shift_by=0)
        x_init, index_to_perturb_dos = partial_dos_attack(x_init)
        index_to_perturb = index_to_perturb_dos + index_to_perturb_section
    else:
        ## 1) Full DOS attack: preferable_extension_amount = 0 and preferable_shift_amount==0
        ## 2) DOS Extension attack: preferable_extension_amount > 0 and preferable_shift_amount ==0
        ## 3) Content shift attack: content_shit=True and preferable_shift_amount > 0

        x_init, index_to_perturb_section = shift_section_by(input,
                                                            preferable_shift_amount=preferable_shift_amount,
                                                            pe_header_start_index_shift_by=0)
        x_init, index_to_perturb_dos = shift_pe_header_by(x_init,
                                                          preferable_extension_amount=preferable_extension_amount)

        """
        solved --> for my understanding: 
        the index_to_perturb in content shift attack should shift preferable_extension_amount,
        since the DOS extension attack has an follow-up impact on content shift attack,
        and first_content_offet does not change/update after DOS_Extension attack
        However, the author said the shifting offset should be the length of index_to_perturb_dos
        --> authors explained this on https://github.com/pralab/secml_malware/issues/11 
        --> after DOS extension, the DOS header is longer, so it spans until pe_position (the old length) + the new one.
        """
        ## shift attack's author's idea
        index_to_perturb = index_to_perturb_dos + [i + len(index_to_perturb_dos) for i in index_to_perturb_section]

    ## remove the perturb index that larger than the input length of detector (fist_n_bytes)
    index_to_perturb = np.array(index_to_perturb)
    index_to_perturb = index_to_perturb[index_to_perturb < max_input_size]
    index_to_perturb = list(index_to_perturb)

    return index_to_perturb, x_init



def get_perturb_index_and_init_x(input: bytearray = None,
                                 preferable_extension_amount: int = 0,
                                 preferable_shift_amount: int = 0x200,
                                 max_input_size: int = 102400,
                                 partial_dos: str='False',
                                 content_shift: str= 'False',
                                 slack: str='False',
                                 combine_w_slack: str='False'):
    """
    get the perturbation index, and initiated x with 0 if new space created by DOS_Extension and Content_Shift
    1) Partial DOS attack: partial_dos = True
    2) Full DOS attack: preferable_extension_amount = 0 and preferable_shift_amount==0
    3) DOS Extension attack: preferable_extension_amount > 0 and preferable_shift_amount ==0
    4) content shift attack: content_shit=True and preferable_shift_amount > 0
    Note: first Content_Shift attack, then DOS_extension attack, to avoid double update start index of pe header

    input: bytearray of raw bytes of pe file
    preferable_extension_amount: number of bytes to extend in DOS_Extension attack. Default as 0 --> full DOS Attack
    preferable_shift_amount:
        - number of bytes to shift first content (i.e., number of space create between PE header and first section)
        - Default as 0x200 (512 bytes)
    max_input_size: -->int: the first n bytes (fixed length) feed into model
    partial_dos: -->bool: true--> only perform partial dos attack
    content_shift: bool: true --> only perform content shift attack
    slack: bool: true --> only perform slack attack
    combine_w_slack: bool: true --> slack attack combine with other attacks

    return: initiated_x:list and index_to_perturb:list
    """
    if partial_dos=='True' and content_shift== 'False' and slack== 'False':
        print(f'partial DOS attack ...')
        ## partial DOS attack
        x_init, index_to_perturb = partial_dos_attack(input)
    elif partial_dos== 'False' and content_shift== 'True' and slack== 'False':
        ## content shit attack
        print('content shift attack ...')
        x_init, index_to_perturb = shift_section_by(input,
                                                    preferable_shift_amount=preferable_shift_amount,
                                                    pe_header_start_index_shift_by=0)
    elif partial_dos== 'False' and content_shift== 'False' and slack== 'True':
        ## slack attack
        print('slack attack ...')
        x_init, index_to_perturb = slack_attack(x=input,max_input_size=max_input_size)
    elif partial_dos== 'True' and content_shift == 'True' and slack== 'False':
        ## content shift attack + partial dos
        print('partial DOS and content shift attack ...')
        x_init, index_to_perturb_section = shift_section_by(input,
                                                            preferable_shift_amount=preferable_shift_amount,
                                                            pe_header_start_index_shift_by=0)
        x_init, index_to_perturb_dos = partial_dos_attack(x_init)
        index_to_perturb = index_to_perturb_dos + index_to_perturb_section
    elif partial_dos== 'True' and content_shift == 'False' and slack== 'True':
        ## partial DOS + slack attack
        print('partial DOS and slack attack ...')
        x_init, index_to_perturb_dos = partial_dos_attack(input)
        x_init, index_to_perturb_slack = slack_attack(x=input, max_input_size=max_input_size)
        index_to_perturb = index_to_perturb_dos + index_to_perturb_slack
    elif partial_dos== 'False' and content_shift == 'True' and slack== 'True':
        ## content shift + slack attack
        print('content shift and slack attack ...')
        x_init, index_to_perturb_section = shift_section_by(input,
                                                            preferable_shift_amount=preferable_shift_amount,
                                                            pe_header_start_index_shift_by=0)
        x_init, index_to_perturb_slack = slack_attack(x=x_init, max_input_size=max_input_size)
        index_to_perturb = index_to_perturb_section + index_to_perturb_slack
    elif partial_dos== 'True' and content_shift == 'True' and slack== 'True':
        ## partial dos + content shift + slack attack
        print('partial dos and content shift and slack attack ...')
        x_init, index_to_perturb_dos = partial_dos_attack(input)
        x_init, index_to_perturb_section = shift_section_by(input,
                                                            preferable_shift_amount=preferable_shift_amount,
                                                            pe_header_start_index_shift_by=0)
        x_init, index_to_perturb_slack = slack_attack(x=x_init, max_input_size=max_input_size)

        index_to_perturb = index_to_perturb_dos + index_to_perturb_section + index_to_perturb_slack
    else:
        ## 1) Full DOS attack: preferable_extension_amount = 0 and preferable_shift_amount==0, combine_w_slack=False
        ## 2) DOS Extension attack: preferable_extension_amount > 0 and preferable_shift_amount ==0,combine_w_slack=False
        ## ...

        if preferable_shift_amount==0 and combine_w_slack=='False':
            if preferable_extension_amount == 0:
                print(f'Full DOS attack ...')
            else:
                print(f'DOS extension attack ...')
            x_init, index_to_perturb_dos = shift_pe_header_by(input,
                                                              preferable_extension_amount=preferable_extension_amount)
            index_to_perturb = index_to_perturb_dos
        elif preferable_shift_amount>0 and combine_w_slack=='False':
            if preferable_extension_amount == 0:
                print('Full DOS + Content Shift attack ...')
            else:
                print('DOS extension + Content Shift attack ...')
            x_init, index_to_perturb_section = shift_section_by(input,
                                                                preferable_shift_amount=preferable_shift_amount,
                                                                pe_header_start_index_shift_by=0)
            x_init, index_to_perturb_dos = shift_pe_header_by(x_init,
                                                              preferable_extension_amount=preferable_extension_amount)
            if preferable_extension_amount == 0:
                index_to_perturb = index_to_perturb_dos + index_to_perturb_section
            else:
                index_to_perturb = index_to_perturb_dos + [i + len(index_to_perturb_dos) for i in
                                                           index_to_perturb_section]
        elif preferable_shift_amount==0 and combine_w_slack=='True':
            if preferable_extension_amount == 0:
                print('Full DOS + Slack attack ...')
            else:
                print('DOS extension + Slack attack ...')
            x_init, index_to_perturb_dos = shift_pe_header_by(input,
                                                              preferable_extension_amount=preferable_extension_amount)
            x_init, index_to_perturb_slack = slack_attack(x=x_init, max_input_size=max_input_size)
            if preferable_extension_amount == 0:
                index_to_perturb = index_to_perturb_dos + index_to_perturb_slack
            else:
                index_to_perturb = index_to_perturb_dos + [i + len(index_to_perturb_dos) for i in
                                                           index_to_perturb_slack]
        elif preferable_shift_amount > 0 and combine_w_slack == 'True':
            if preferable_extension_amount == 0:
                print('Full DOS + Content Shift + Slack attack ...')
            else:
                print('DOS extension + Content Shift + Slack attack ...')
            x_init, index_to_perturb_section = shift_section_by(input,
                                                                preferable_shift_amount=preferable_shift_amount,
                                                                pe_header_start_index_shift_by=0)
            x_init, index_to_perturb_slack = slack_attack(x=x_init, max_input_size=max_input_size)
            x_init, index_to_perturb_dos = shift_pe_header_by(x_init,
                                                              preferable_extension_amount=preferable_extension_amount)
            if preferable_extension_amount == 0:
                index_to_perturb = index_to_perturb_dos + index_to_perturb_section +index_to_perturb_slack
            else:
                index_to_perturb = index_to_perturb_dos + [i + len(index_to_perturb_dos) for i in index_to_perturb_section]\
                                   +[i + len(index_to_perturb_dos) for i in index_to_perturb_slack]

    ## remove the perturb index that larger than the input length of detector (fist_n_bytes)
    index_to_perturb = np.array(index_to_perturb)
    index_to_perturb = index_to_perturb[index_to_perturb < max_input_size]
    index_to_perturb = list(index_to_perturb)

    return index_to_perturb, x_init
