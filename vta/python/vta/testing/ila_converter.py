# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Converts a VTA simulator JSON dump into an ILA instruction trace"""
import ctypes
import json
import math

VIR_MEM_MODES = {
    'INP': 1,
    'WGT': 2,
    'BIAS': 3,
    'UOP': 4
}

LITTLE_ENDIAN = True

def hex_string_to_bytes(hex_string):
    digits = hex_string.split('x')[1]
    assert len(hex_string) % 2 == 0
    bytes = [
        '0x{}'.format(digits[2*i:2*(i+1)])
        for i in range(len(hex_string) / 2)
    ]
    # if it's a little-endian integer,
    # the last digits are the first byte
    if LITTLE_ENDIAN:
        bytes = bytes[::-1]
    return bytes


def set_bits(bit_array, value, num_bits, idx):
    # assumes value to be given as a hex string, so
    # we will convert each byte
    val_bytes = hex_string_to_bytes(value)

    bits = ''
    for byte in val_bytes:
        bits += format(int(byte, 16), '08b')

    # use only the last num_bits if we have more than that
    if len(bits) > num_bits:
        bits = bits[len(bits) - num_bits:]
    # add leading 0's if we didn't have enough
    if len(bits) < num_bits:
        bits = ('0'*(num_bits - len(bits))) + bits
    for i, bit in enumerate(bits):
        bit_array[idx + i] = bit
    return idx + num_bits


def reconstitute_mem_insn(
        opcode, pop_prev, pop_next, push_prev, push_next,
        memory_type, sram_base, dram_base, y_size,
        x_size, x_stride, y_pad_0, y_pad_1, x_pad_0, x_pad_1):
    # given field values for a memory instruction,
    # produces a byte array encoding that instruction;
    # this is to enable having to edit dram addresses
    # during the translation if we have to

    # (use bin to assemble a big binary blob of 128 bits)
    bits = ['0'] * 128
    next_idx = set_bits(bits, opcode, 3, 0)
    next_idx = set_bits(bits, pop_prev, 1, next_idx)
    next_idx = set_bits(bits, pop_next, 1, next_idx)
    next_idx = set_bits(bits, push_prev, 1, next_idx)
    next_idx = set_bits(bits, push_next, 1, next_idx)
    next_idx = set_bits(bits, memory_type, 2, next_idx)
    next_idx = set_bits(bits, sram_base, 16, next_idx)
    next_idx = set_bits(bits, dram_base, 32, next_idx)
    next_idx = set_bits(bits, y_size, 16, next_idx)
    next_idx = set_bits(bits, x_size, 16, next_idx)
    next_idx = set_bits(bits, x_stride, 16, next_idx)
    next_idx = set_bits(bits, y_pad_0, 4, next_idx)
    next_idx = set_bits(bits, y_pad_1, 4, next_idx)
    next_idx = set_bits(bits, x_pad_0, 4, next_idx)
    next_idx = set_bits(bits, x_pad_1, 4, next_idx)

    # we assembled the bits in order so there is no concern about endianness here
    combine_bits = ''.join(bits)
    num_bytes = 16
    assert len(combine_bits)/8 == num_bytes
    ret = [
        format(int(combine_bits[i*8:(i+1)*8], 2), '#04x')
        for i in range(num_bytes)
    ]
    return ret


def ila_instruction(
        insn_idx=0, instr_in='0x00', mem_addr=0,
        mem_bias_in='0x00', mem_inp_in='0x00',
        mem_mode=0,
        mem_uop_in='0x00', mem_wgt_in='0x00'):
    # helper function for filling out fields in an ILA instruction
    return {
        'instr No.': insn_idx,
        'instr_in': instr_in,
        'mem_addr': mem_addr,
        'mem_bias_in': mem_bias_in,
        'mem_inp_in': mem_inp_in,
        'mem_mode': mem_mode,
        'mem_uop_in': mem_uop_in,
        'mem_wgt_in': mem_wgt_in
    }


def create_ila_dram_insn(target, addr, data, insn_idx):
    if target not in VIR_MEM_MODES:
        raise Exception(f'Invalid target: {target}')

    vir_mem_mode = VIR_MEM_MODES[target]

    # read in a single byte xx in hex, present as
    # 0xffffffxx
    if data == '0xXX':
        raise Exception(f'Attempting to write padding value at addr {addr}')

    # in principle, we should only set the data field we plan to write;
    # however, if we use the same one in every field, that is sufficient
    return ila_instruction(
        insn_idx=insn_idx, mem_mode=vir_mem_mode,
        mem_addr=addr,
        mem_bias_in=data, mem_inp_in=data,
        mem_uop_in=data, mem_wgt_in=data)


def create_ila_vta_insn(insn_bytes, insn_idx):
    # we combine the instruction bytes into a single 128-bit integer
    digit_strs = [byte.split('x')[1] for byte in insn_bytes]
    # if the integer is little endian, then the first bytes are the last digits
    if LITTLE_ENDIAN:
        digit_strs = digit_strs[::-1]
    insn_str = '0x{}'.format(''.join(digit_strs))
    return ila_instruction(insn_idx=insn_idx,
                           instr_in=insn_str)


def generate_dram_insns(sim_dump, insn_idx):
    """
    Generates VTA 'helper' instructions for setting up
    the DRAM.

    Parameters
    ----------
    sim_dump : dict[str, any]
        JSON dump from simulator
    insn_idx : int
        Starting instruction number

    Returns
    -------
    List of ILA instructions setting up the DRAM
    """
    dram_dumps = sim_dump['dram']
    ret = []
    for dump in dram_dumps:
        mem_type = dump['context']
        # the ILA does not need a dump of the instructions
        if mem_type == 'INSN':
            continue
        if mem_type == 'unknown':
            raise Exception('Unknown memory type encountered')
        addr = int(dump['start_addr'], 16)
        for i, byte in enumerate(dump['bytes']):
            if byte == '0xXX':
                continue
            offset_addr = addr + i
            ret.append(create_ila_dram_insn(
                mem_type, format(offset_addr, '#010x'), byte, insn_idx))
            insn_idx += 1
    return ret


def convert_prog_insns(sim_dump, insn_idx):
    """
    Converts the VTA instructions in the simulator dump
    to the corresponding ILA instructions.

    Parameters
    ----------
    sim_dump : dict[str, any]
        JSON dump from simulator
    insn_idx : int
        Starting instruction number

    Returns
    -------
    List of ILA instructions corresponding to the
    VTA instructions in the dump
    """
    insns = sim_dump['insns']
    ret = []
    for insn in insns:
        ret.append(create_ila_vta_insn(insn['raw_bytes'], insn_idx))
        insn_idx += 1
    return ret


def convert(src_path, dest_path):
    """
    Convert simulator JSON dump into an ILA program fragment.

    Parameters
    ----------
    src_path : str
        Path to simulator JSON dump
    dest_path : str
        Path at which corresponding ILA program fragment should be written
    """
    with open(src_path, 'r') as f:
        source = json.load(f)

    memory_insns = generate_dram_insns(source, 0)
    prog_insns = convert_prog_insns(source, len(memory_insns))

    prog_frag = {'program_fragment': memory_insns + prog_insns}

    with open(dest_path, 'w') as f:
        json.dump(prog_frag, f, indent=4)
