import os
import subprocess
import re

def call_python(fname, args=[]):
    subprocess.check_call(["python", "-u", fname] + args)

class Files:

    @classmethod
    def delete_pyc(cls, folder):
        for root, dirs, files in os.walk(folder):
            for name in files:
                if re.match(".*\.pyc", name):
                    os.remove(os.path.join(root, name))
    
    @classmethod
    def delete_files(cls, folder):
        # delete files
        for root, dirs, files in os.walk(folder):
            for name in files:
                os.remove(os.path.join(root, name))
    
    @classmethod
    def mkdir(cls, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


class Writer:
    @classmethod
    def create_empty(cls, path):
        f = open(path, 'w')
        f.close()
    
    @classmethod
    def append_line(cls, path, content):
        with open(path, 'a') as f:
            f.write(content + '\n')
    
    @classmethod
    def append_line_list(cls, path, lst, sep="\t"):
        cls.append_line(path, sep.join([str(e) for e in lst]))

class ModelSetting:

    def __init__(self, model_type, model_params, feature_set, other="x"):
        self.model_type = model_type
        self.model_params = model_params
        self.feature_set = feature_set
        self.other = other

    def name(self):
        return ".".join((self.model_type, self.model_params, self.feature_set, self.other))
