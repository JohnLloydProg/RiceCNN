from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
import pandas as pd
import keras
import settings
import numpy as np
import random
import time
import os

model = keras.models.load_model('./model')

data = []
features = []
labels = []
x_data = []
y_data = []
batch_size = 20

data.clear()
df = pd.read_excel(f'./data_template.xlsx', header=0, index_col=0)
base_dir = f'./raw_data/'
if f'data.xlsx' not in os.listdir('.'):
    print('Preparing Data')

    for category in  os.listdir(base_dir):
        path = os.path.join(base_dir, category) + '/'
        for file in os.listdir(path):
            data.append((os.path.join(path, file), settings.CATEGORIES.index(category)))
    data = data[:600]
    random.shuffle(data)
    rep = len(data) // batch_size

    features.clear()
    labels.clear()
    for feature, label in data:
        features.append(feature)
        labels.append(label)

    print('Starting Evaluation')

    for j in tqdm(range(rep)):
        start = time.time()
        x_data = np.array([resize(imread(feature), (settings.picture_height, settings.picture_width, 3)) for feature in features[j*batch_size:(j+1)*batch_size]])
        y_data = np.array(labels[j*batch_size:(j+1)*batch_size])
        results = model.evaluate(x_data, y_data)
        end = time.time()
        exe_time = end - start
        df.loc[j] = (f'Batch {j}', results[1], exe_time)
        
    print('Exporting Data')
    df.to_excel(f'./data.xlsx')
    x_data = []
    y_data = []
    