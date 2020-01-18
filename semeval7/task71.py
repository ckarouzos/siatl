import csv
import os
import re
from zipfile import ZipFile

import shutil
import urllib
import urllib.request

from torch.utils.data import Dataset
from sklearn.utils import shuffle
import numpy as np

TASK7_URL = 'https://www.cs.rochester.edu/u/nhossain/humicroedit/semeval-2020-task-7-data.zip'  # noqa

def safe_mkdirs(path: str) -> None:
    """! Makes recursively all the directory in input path """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception:
            raise IOError(
                (f"Failed to create recursive directories: {path}"))

def download_url(url: str, dest_path: str) -> str:
    """
    Download a file to a destination path given a URL
    """
    name = url.rsplit('/')[-1]
    dest = os.path.join(dest_path, name)
    safe_mkdirs(dest_path)
    response = urllib.request.urlopen(url)
    with open(dest, 'wb') as fd:
        shutil.copyfileobj(response, fd)
    return dest

class Task1Dataset(Dataset):
    def __init__(self, directory, transforms=[], train=True):
        dest = download_url(TASK7_URL, directory)
        with ZipFile(dest, 'r') as zipfd:
            zipfd.extractall(directory)
        split = 'train' if train else 'dev'
        self._file = os.path.join(directory, 'data', 'task-1', f'{split}.csv')
        self.transforms = transforms
        self.train = train
        if train:
            (self.ids,
             self.original,
             self.edit,
             self.grades,
             self.mean_grade) = self.get_metadata(self._file)
        else:
            (self.ids,
             self.original,
             self.edit) = self.get_metadata_dev(self._file)

    def get_metadata(self, _file):
        rows = []
        with open(_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                rows.append(row)

        rows = rows[1:]  # strip header
        ids, original, edit, grades, mean_grade = zip(*rows)
        return ids, original, edit, grades, mean_grade

    def get_metadata_dev(self, _file):
        rows = []
        with open(_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                rows.append(row)

        rows = rows[1:]  # strip header
        ids, original, edit = zip(*rows)
        return ids, original, edit

    def _edit(self, original, edit):
        return re.sub(r'<.*?/>', f'<{edit}/>', original)

    def map(self, t):
        self.transforms.append(t)
        return self

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        original = self.original[idx]
        edit = self.edit[idx]
        ids = self.ids[idx]
        # grades = self.grades[idx]
        edited = self._edit(original, edit)
        if self.train:
            mean_grade = self.mean_grade[idx]

        for t in self.transforms:
            original = t(original)
            edited = t(edited)

        # grades = [int(g) for g in str(grades)]
        if self.train:
            return original, edited, float(mean_grade)
        else:
            return ids, original, edited

def make_Xy(test_size=0.2, shuffle=True):
    data = Task1Dataset('../data')
    X = []
    y= []
    for d in data:
        original, edited, grade = d
        text = original + edited
        X.append(text)
        y.append(grade)
    size = len(X)
    test_split = int(np.floor(test_size* size))
    if shuffle:
        X, y = shuffle(X, y, random_state=0)
    X_train = X[test_split:]
    X_test = X[:test_split]
    y_train = y[test_split:]
    y_test = y[:test_split]
    dataset =[X_train, y_train, X_test, y_test]
    return dataset

if __name__ == '__main__':
    # a dummy data loading!
    data = Task1Dataset('../data/', train=False)
    for d in data:  # type: ignore
        print(d)
