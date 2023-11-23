import datetime
import argparse
import os
import shutil
from pathlib import Path
def Copy_Add_Timestamp(args):
    output_dir = Path(args.copy_folder)
    input_file = Path(args.his_filename_input)
    
    timestring = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    output_filename = f'{timestring}_{input_file.name}'
    
    os.makedirs(Path(args.copy_folder), exist_ok=True)
    shutil.copy(args.his_filename_input, output_dir / output_filename )
    return

parser = argparse.ArgumentParser("Take Hisfile, copy it to another location with time as a postfix to the file")
parser.add_argument('--his_filename_input', help="Path to His-file where a rolling average window should be applied")
parser.add_argument('--copy_folder', help="Path to where averaged His-file should be written away, ")

if __name__ == '__main__':
    args = parser.parse_args()
    Copy_Add_Timestamp(args)