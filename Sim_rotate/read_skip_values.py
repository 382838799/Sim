import numpy as numpy
import ast



def read_skip_values(filename):
    skip_values = []
    with open(filename, 'r') as f:
        for line in f:
            parsed_list = ast.literal_eval(line.strip())

            if parsed_list:
                skip_values.extend(parsed_list)

    return skip_values