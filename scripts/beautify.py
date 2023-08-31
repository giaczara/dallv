from argparse import ArgumentParser
from shutil import move

parser = ArgumentParser()
parser.add_argument("script", type=str)
args = parser.parse_args()

TEMPFILE = "temp.sh"

input_stream = open(args.script, "r")
output_stream = open(TEMPFILE, "w")
for line in input_stream:
    if (
        line.startswith("#")
        or line.startswith("\n")
        or line.startswith("eval")
        or line.startswith("conda")
        or line.startswith("export")
    ):
        output_stream.write("{}".format(line))
    else:
        split_line = line.split(" ")
        arg_count = 0
        for arg in split_line:
            if arg_count == 0:
                output_stream.write("CUDA_VISIBLE_DEVICES=0 python -W ignore {} \\\n".format(arg))
                arg_count += 1
            elif arg_count == len(split_line) - 1:
                output_stream.write("  {} \n".format(arg))
            else:
                output_stream.write("  {} \\\n".format(arg))
                arg_count += 1

input_stream.close()
output_stream.close()

move(TEMPFILE, args.script)
