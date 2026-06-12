# save as check_setup.py and run it
import sys
print(f"Python: {sys.version}")

import sklearn;    print(f"scikit-learn : {sklearn.__version__}")
import pandas;     print(f"pandas       : {pandas.__version__}")
import numpy;      print(f"numpy        : {numpy.__version__}")
import tensorflow; print(f"tensorflow   : {tensorflow.__version__}")
import pefile;     print(f"pefile       : OK")
import capstone;   print(f"capstone     : OK")

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPU detected  : {'✅ YES — ' + gpus[0].name if gpus else '❌ No GPU found'}")

print("\n✅ All libraries installed successfully!")