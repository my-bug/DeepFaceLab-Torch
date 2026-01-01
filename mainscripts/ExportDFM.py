import os
import sys
import traceback
import queue
import threading
import time
import numpy as np
import itertools
from pathlib import Path
from core import pathex
from core import imagelib
import cv2
import models
from core.interact import interact as io


def main(model_class_name, saved_models_path):
    silent = str(os.environ.get('DFL_SILENT_INPUT', '')).strip().lower() in ('1', 'y', 'yes', 'true', 'on')

    model = models.import_model(model_class_name)(
                        is_exporting=True,
                        saved_models_path=saved_models_path,
                        cpu_only=True,
                        silent_start=silent,
                        force_model_name='export' if silent else None)
    model.export_dfm () 
