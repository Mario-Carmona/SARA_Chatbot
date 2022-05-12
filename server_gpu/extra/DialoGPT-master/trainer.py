
import logging
import os
import subprocess as sp
import sys






logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO
)
logger = logging.getLogger(__name__)



target_folder = "/mnt/homeGPU/mcarmona/microsoft/DialoGPT-medium"

PYTHON_EXE = 'python'
PROJECT_FOLDER = os.path.dirname(os.path.realpath(__file__))
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, 'models')
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')

if os.path.exists(MODEL_FOLDER):
    print(f'Found existing models folder at {MODEL_FOLDER}, skip creating a new one!')
    os.makedirs(MODEL_FOLDER, exist_ok=True)
else:
    os.makedirs(MODEL_FOLDER)


#########################################################################
# Prepare Data
#########################################################################

logger.info('Preparing Data...')
data_path = os.path.join(DATA_FOLDER, 'dataset_train.tsv')
MAX_LEN = 128
data_db = f'{data_path[:-4]}.{MAX_LEN}len.db'
if os.path.isdir(data_db):
    print(f'{data_db} exists, skip prepro.py')
else:
    cmd = ['prepro.py', '--corpus', data_path, '--max_seq_len', f'{MAX_LEN}']
    cmd = ' '.join(cmd) #% {'CODE_ROOT': CODE_ROOT}
    print(cmd)
    ret = sp.run([PYTHON_EXE] + cmd.split(' '), stdout=sp.PIPE, stderr=sp.STDOUT, cwd=PROJECT_FOLDER)
    if ret.returncode != 0:
        print(f'error occurred, {ret.stdout}')
        sys.exit(ret.returncode)
logger.info('Done!\n')




#########################################################################
# Train !
#########################################################################
logger.info('Generating training CMD!')
logger.info('If there is any problem, please copy (modify) and run command below')
logger.info('#########################################################################')
train_cmd = 'LSP_train.py'
args = [
    '--model_name_or_path', target_folder,
    '--init_checkpoint', os.path.join(target_folder, 'pytorch_model.bin'),
    '--train_input_file', data_db ,  # file from last step
    '--eval_input_file', './data/dataset_valid.tsv',   # dummy test data
    '--output_dir', os.path.join(MODEL_FOLDER, 'output_model'),
    '--seed', '0',
    '--max_seq_length', '128',
    '--train_batch_size', '512',
    '--gradient_accumulation_steps', '8',
    '--eval_batch_size', '64',
    '--learning_rate', '1e-5',
    '--init_epoch', '0',
    '--num_epochs', '100',
    '--warmup_steps', '4000',
    '--normalize_data', 'true',
    '--fp16', 'true',
    '--lr_schedule', 'noam',
    '--loss_scale', '0.0',
    '--no_token_id', 'true',
    '--pbar', 'true'
]

arg = ' '.join(args)
train_cmd = train_cmd + ' ' + arg
print(PYTHON_EXE + ' ' +train_cmd)
logger.info('#########################################################################')
with open('./output.log', 'wb') as f: 
    process = sp.Popen([PYTHON_EXE] + train_cmd.split(' '), stdout=sp.PIPE, stderr=sp.STDOUT, cwd=PROJECT_FOLDER)
    for line in iter(process.stdout.readline, b''): 
        sys.stdout.write(line.decode(sys.stdout.encoding)) 
        f.write(line)
logger.info('Done!\n')