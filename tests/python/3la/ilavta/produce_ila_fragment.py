from ila_converter import convert
import sys

if __name__ == '__main__':
    dump_file = 'vta_sim_dump.json'
    dest_file = 'ila_vta_fragment_input.json'
    if len(sys.argv) > 1:
        dump_file = sys.argv[1]
    if len(sys.argv) > 2:
        dest_file = sys.argv[2]
    convert(dump_file, dest_file)
