import sys
sys.path += ["utils"]
import util
from util import Files
from config import Config
config = Config("production", "v01t", test=True)
config.dump()

Files.delete_pyc(".")

def feat_run_all():
    util.call_python("feat/create_id.py")
    util.call_python("feat/preprocess.py")
    util.call_python("feat/feat_MKsum.py")
    util.call_python("feat/feat_PI2.py")
    util.call_python("feat/feat_basic.py")
    util.call_python("feat/feat_keras.py")
    print "feat finished"

feat_run_all()
util.call_python("model/model_run_main.py")
