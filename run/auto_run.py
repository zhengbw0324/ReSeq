import os

python = 'python' #Your virtual environment path
format_str=python+' ../main.py --model={} --dataset={} --config_files={}'
model='ReSeq'
dataset='ask'
config_file= 'config/ask.yaml'

if os.system(format_str.format(model,dataset,config_file)):
        raise ValueError('Error '+model)
