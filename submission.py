import os
import shutil
import json

src_dir = './src'
submission_dir = './submission'

with open('./studentID.json', 'r') as f:
    json_data = json.load(f)
    student_id = json_data['id']

file_list = os.listdir(src_dir)

for i in file_list:
    if i.endswith('.ipynb'):
        assignName = i.split('.')[0]
        newFileName = f'aip2_{assignName}_{student_id}.ipynb'
        shutil.copy(src_dir + '/' + i, submission_dir + '/' + newFileName)
