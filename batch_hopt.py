import sys
sys.path += ["utils"]
import util
from util import Files
from config import Config
config = Config("another", "v01a", test=True)
config.dump()

Files.delete_pyc(".")

def feat_run_all():
    util.call_python("feat/create_id.py")
    util.call_python("feat/preprocess.py")
    util.call_python("feat/feat_basic.py")
    util.call_python("feat/feat_keras.py")
    util.call_python("feat/feat_MKsum.py")
    util.call_python("feat/feat_PI2.py")
    print "feat finished"

feat_run_all()
util.call_python("others/hopt_run_xgb.py")
util.call_python("others/hopt_run_ert.py")
util.call_python("others/hopt_run_vw.py")
util.call_python("others/hopt_run_keras.py")
