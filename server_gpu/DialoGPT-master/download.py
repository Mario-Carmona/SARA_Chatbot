
import os
from functools import partial
from demo_utils import download_model_folder



PROJECT_FOLDER = os.path.dirname(os.path.realpath(__file__))
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, 'models')

download_model = partial(download_model_folder, DATA_FOLDER=MODEL_FOLDER)
target_folder = download_model(model_size='medium', dataset='dstc', from_scratch=False)




