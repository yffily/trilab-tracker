#! /usr/bin/env python3
import sys
from trilabtracker import start_gui

if __name__=='__main__':
    input_dir = sys.argv[1] if len(sys.argv)>1 else None
    sys.exit(start_gui(input_dir))

