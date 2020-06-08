import os, time
from subprocess import Popen, PIPE, STDOUT
from src.Config import Config
import numpy as np

FIELD = ['SUMANDAK', 'SUMANDAK TENGAH', 'SUMANDAK SELATAN', 'SUMANDAK TEPI']
STRINGS = [] # "Sumandak-A001-TS", "Sumandak-B008-TS"]
STRING_SUFFIX = '_SMDK'
DART_BOOL = True
ALGORITHMS = ["RFR", "XGBR", "LGBMR"] # list of tuned models [RFR_tuned XGBR_tuned LGBMR_tuned]
PREDVAR = "SAND_COUNT"
PREDFILE = "pred1"

CMDs = [
    # r"app.py download -r {0} -f {1} -s {2}".format(DART_BOOL, FIELD, STRING_SUFFIX)
]

# path = os.path.join(Config.FILES["DATA"], Config.FILES["MODELS"], PREDVAR+"_test.ids")

CMDs.extend([
    r"app.py analysis -r -s {0}".format(STRING_SUFFIX),
    r"app.py train -r -a {0} -s {1}".format(' '.join(ALGORITHMS), STRING_SUFFIX),
    r"app.py predict -r -s {0}".format(STRING_SUFFIX),
    # r"app.py report -r -s {0}".format(STRING_SUFFIX)
])

def main():

    for cmd in CMDs:
        now = time.time()

        try: 
            command = Popen('python {}'. format(cmd), stdout =PIPE, stderr = STDOUT, universal_newlines=True, shell=True)
            result = command.communicate()[0]
            print(result)

            #result = subprocess.check_output('python {}'. format(cmd), stderr = subprocess.PIPE, shell=True)
            runtime = time.time() - now
            print("The task '{}' has been successfully executed within {:.2f} seconds".format(cmd, runtime))
        except Exception as e:
            runtime = time.time() - now
            print("Error in performing task '{}' within {:.2f} seconds". format(cmd, runtime))

if __name__ == '__main__':
    main()

