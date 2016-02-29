from model_run import run_model_all
from util import ModelSetting

# model settings
# model setting - model_type, model_param, feature_set
model_settings = []
model_settings.append(ModelSetting("xgb", "xgb_hopt0", "f1"))
model_settings.append(ModelSetting("ert", "ert_hopt0", "fnn1"))
model_settings.append(ModelSetting("vw", "vw_hopt0", "fvw1"))
model_settings.append(ModelSetting("keras", "keras_hopt0", "fnn1"))

model_settings_fin = []
model_settings.append(ModelSetting("xgb", "xgb_hopt0", "f1_en1"))
model_settings.append(ModelSetting("xgb", "xgb_hopt0", "en1"))

if __name__ == "__main__":
    run_model_all(model_settings, False)
    run_model_all(model_settings_fin, True)
