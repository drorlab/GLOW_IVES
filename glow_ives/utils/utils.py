import subprocess
import pandas as pd

def run_cmd(cmd, error_msg=None, raise_except=False):
    try:
        return subprocess.check_output(
            cmd,
            universal_newlines=True,
            shell=True)
    except Exception as e:
        print(e)
        if error_msg is not None:
            print(error_msg)
        if raise_except:
            raise e
