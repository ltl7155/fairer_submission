import os
import zipfile
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
import shutil
import pickle

labels_path = 'data/CelebA/list_attr_celeba.txt'
image_path = 'data/CelebA/img_align_celeba/'
split_path = 'data/CelebA/list_eval_partition.txt'

labels_df = pd.read_csv(labels_path)

label_dict = {}
for i in range(1, len(labels_df)):
    label_dict[labels_df['202599'][i].split()[0]] = [x for x in labels_df['202599'][i].split()[1:]]

label_df = pd.DataFrame(label_dict).T
label_df.columns = (labels_df['202599'][0]).split()
label_df.replace(['-1'], ['0'], inplace = True)

print(label_df)

# generate train/val/test
files = glob(image_path + '*.jpg')

split_file = open(split_path, "r")
lines = split_file.readlines()

# os.mkdir('celeba/tmp/')
# for i in ['train', 'val', 'test']:
#     os.mkdir(os.path.join('celeba/tmp/', i))

train_file_names = []
train_dict = {}
valid_file_names = []
valid_dict = {}
test_file_names = []
test_dict = {}
for i in tqdm(range(len(lines))):
# for i in tqdm(range(2)):
    file_name, sp = lines[i].split()
    sp = sp.split('\n')[0]
    if sp == '0':
        labels = np.array(label_df[label_df.index==file_name])
        train_dict[file_name] = labels
        train_file_names.append(file_name)
        source_path = image_path + file_name
#         shutil.copy2(source_path, os.path.join('celeba/tmp/train', file_name))
    elif sp == '1':
        labels = np.array(label_df[label_df.index==file_name])
        valid_dict[file_name] = labels
        valid_file_names.append(file_name)
        source_path = image_path + file_name
#         shutil.copy2(source_path, os.path.join('celeba/tmp/val', file_name))
    else:
        labels = np.array(label_df[label_df.index==file_name])
        test_dict[file_name] = labels
        test_file_names.append(file_name)
        source_path = image_path + file_name
#         shutil.copy2(source_path, os.path.join('celeba/tmp/test', file_name))

print(train_dict.values())
# print(np.array(list(train_dict.values()))[:, 0])
# train_df = pd.DataFrame(train_dict.values())
#     train_df[fname] = train_dict[fname]
# train_df.index = train_file_names
# train_df.columns = ['labels']

train_data = {"fname": train_file_names, "labels": train_dict.values()}
train_df = pd.DataFrame(train_data)
train_df = train_df.set_index("fname")
train_df.index.name=None
# print(train_df.index)
# print(train_df)


valid_data = {"fname": valid_file_names, "labels": valid_dict.values()}
valid_df = pd.DataFrame(valid_data)
valid_df = valid_df.set_index("fname")
valid_df.index.name=None

test_data = {"fname": test_file_names, "labels": test_dict.values()}
test_df = pd.DataFrame(test_data)
test_df = test_df.set_index("fname")
test_df.index.name=None

df = {}
df['train'] = train_df
df['val'] = valid_df
df['test'] = test_df
with open('celeba/data_frame.pickle', 'wb') as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('data frame saved')
