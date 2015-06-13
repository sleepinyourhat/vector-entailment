from __future__ import with_statement
import subprocess
from subprocess import check_output
import glob


def grep_lines(filename, query):
    return subprocess.check_output(["/bin/grep", query, filename])

listing = glob.glob(
    '/scr/sbowman/tuning6-e-and-or-deep*')

for filename in listing:
    print grep_lines(filename + "/example_log", "lineLimit")
