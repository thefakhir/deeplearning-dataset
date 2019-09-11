#  Copyright (c) 2019.
#  @author Fakhir Khan
#  @part of Vision Classifier for SlashNext

import copy
import logging
import os

import h5py
import numpy as np
import pandas as pd
import psutil
from PIL import Image
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class Dataset:

    def __init__(self, input_arguments):

        if 'debug' in input_arguments.keys():
            self.debug_enabled = input_arguments['debug']
        else:
            self.debug_enabled = False

        assert 'train_csv_path' in input_arguments.keys(), 'Training Csv file was not specified in the input arguments'
        assert os.path.isfile(input_arguments['train_csv_path']), 'Training csv file was not found.'

        self.train_csv_path = input_arguments['train_csv_path']
        if 'test_csv_path' in input_arguments.keys():
            if input_arguments['test_csv_path'] is not None:
                if os.path.isfile(input_arguments['test_csv_path']):
                    self.test_csv_path = input_arguments['test_csv_path']
                else:
                    self.test_csv_path = None
                    logging.warning('Test CSV file was not found. Test Data would be randomly sampled from Train CSV')
            else:
                self.test_csv_path = None
                logging.warning('Test CSV path was not provided. Test Data would be randomly sampled from Train CSV')
        else:
            self.test_csv_path = None
            logging.warning('Test CSV path was not provided. Test Data would be randomly sampled from Train CSV')

        if 'load_chunk' in input_arguments.keys():
            self.load_chunk = input_arguments['load_chunk']
        else:
            self.load_chunk = False

        if 'batch_size' in input_arguments.keys():
            self.batch_size = input_arguments['batch_size']
        else:
            self.batch_size = 8

        assert 'image_shape' in input_arguments.keys(), 'image_shape was not passed to the input_arguments for the Dataset Class'
        self.image_shape = input_arguments['image_shape']

        assert 'base_path' in input_arguments.keys(), 'Base path directory was not provided for loading images from the dataset'
        self.base_path = input_arguments['base_path']
        assert os.path.isdir(
            self.base_path), 'Base path directory was not found. Please check the base_path passed to the Dataset Class.' \
                             '\nBase_path = {0}'.format(input_arguments['base_path'])

        if 'use_train_only' in input_arguments.keys():
            self.use_train_only = input_arguments['use_train_only']
        else:
            self.use_train_only = False

        if 'data_folder_path' in input_arguments.keys():
            self.data_folder_path = input_arguments['data_folder_path']
            if self.use_train_only:
                _flags_for_hd5 = '{0}_trainOnly'.format(self.image_shape[0])
            else:
                _flags_for_hd5 = '{0}'.format(self.image_shape[0])

            self.hd5_files_path = {
                'train': os.path.join(self.data_folder_path, 'train_set_{0}.hd5'.format(_flags_for_hd5)),
                'validation': os.path.join(self.data_folder_path,
                                           'validation_set_{0}.hd5'.format(_flags_for_hd5)),
                'test': os.path.join(self.data_folder_path, 'test_set_{0}.hd5'.format(_flags_for_hd5))}

        else:
            self.data_folder_path = None
            logging.WARNING('No Data folder path was passed to the Dataset Init. HD5 files would not be saved')
            self.hd5_files_path = dict()

        self.dataset_csv, self.classes = self.load_dataset_csv()
        self.dataset_loaded_into_memory = None
        self.dataset_batch_generators = None

    def load_dataset_csv(self):

        train_df = pd.read_csv(self.train_csv_path,
                               delimiter=',',
                               header=0,
                               names=['index', 'label', 'filename'])
        if self.load_chunk:
            train_df = train_df.sample(n=50000,
                                       random_state=42)
        _validation_set_split = 0.1

        if self.test_csv_path is not None:
            logging.info('Loading Test Data from  = TEST csv')
            test_df = pd.read_csv(self.test_csv_path,
                                  delimiter=',',
                                  header=0,
                                  names=['index', 'label', 'filename'])
        else:
            logging.info('Loading Test Data from  = TRAIN csv')
            if not self.use_train_only:
                _test_set_split = 0.2
            else:
                _test_set_split = 50
                _validation_set_split = 5000
                logging.warning('Test Set is not used for the training')
            train_df, test_df = train_test_split(train_df,
                                                 test_size=_test_set_split,
                                                 random_state=42)
        if self.debug_enabled:  # If debugging, use a smaller chunk of data
            train_df = train_df.head(500)
            test_df = test_df.head(100)

        x_train, y_train = train_df['filename'].values, train_df['label'].values
        x_test, y_test = test_df['filename'].values, test_df['label'].values

        x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                          y_train,
                                                          test_size=_validation_set_split,
                                                          random_state=42)
        # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.1, random_state = 42)

        y_train_categorical, _classes = self.convert_to_categorical_labels(y_train)
        y_val_categorical, _ = self.convert_to_categorical_labels(y_val)
        y_test_categorical, classes_test = self.convert_to_categorical_labels(y_test)
        assert classes_test.all() == _classes.all(), "Different classes used in the train and test sets" + \
                                                     '\n Train Classes = ' + str(_classes) + \
                                                     "\n Test Classes = " + str(classes_test)

        train = {'x': x_train, 'y': y_train_categorical}
        validation = {'x': x_val, 'y': y_val_categorical}
        test = {'x': x_test, 'y': y_test_categorical}

        _datasets = {'train': train, 'validation': validation, 'test': test}
        return _datasets, _classes

    @staticmethod
    def convert_to_categorical_labels(y):
        y_categorical = pd.get_dummies(y)
        classes = y_categorical.columns.values
        return y_categorical.values, classes

    def load_all_batches(self, x, y):
        batch_images_y1 = np.array([label for label in y])
        batch_images_x = np.array(
            [img_to_array(
                (load_img(os.path.join(self.base_path, image_name)).resize(self.image_shape, Image.ANTIALIAS)))
                for image_name in tqdm(x)])

        batch_images_x /= 255
        return batch_images_x, batch_images_y1

    def get_steps(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        _steps = dict()
        _prod = 1.0
        for chunk in self.dataset_csv:
            _length_of_data = self.dataset_csv[chunk]['y'].shape[0]
            _steps[chunk] = _length_of_data // batch_size
            _prod *= float(_steps[chunk])
        assert _prod > 0, 'The batch size is too big for creating a single batch in a dataset chunk'
        return _steps

    def load_data_into_memory(self):
        if self.can_load_data_into_memory():
            if self.check_hd5_files():
                _dataset_in_memory = self.read_hd5_files()
            else:
                _dataset_in_memory = dict()
                for chunk in self.dataset_csv:
                    logging.warning('Loading {0} data into memory'.format(str(chunk)))
                    _x, _y = self.load_all_batches(x=self.dataset_csv[chunk]['x'],
                                                   y=self.dataset_csv[chunk]['y'])

                    _dataset_in_memory[chunk] = {'x': -_x, 'y': _y}
                # print('Size of Training Data = {0}'.format(len(_dataset_in_memory['train']['x'])))
                # print('Size of Validation Data = {0}'.format(len(_dataset_in_memory['validation']['x'])))
                # print('Size of Test Data = {0}'.format(len(_dataset_in_memory['test']['x'])))

                self.save_hd5_files(dataset_dict=_dataset_in_memory)
            return _dataset_in_memory
        else:
            raise RuntimeError('Data could not be loaded into memory. This may cause issues in running the system')

    def load_data_into_generators(self, shuffle=False, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        _batch_generators_dict = dict()
        for chunk in self.dataset_csv:
            _dataset_chunk = self.dataset_csv[chunk]
            _batch_generators_dict[chunk] = self.batch_generator(data_x=_dataset_chunk['x'],
                                                                 data_y=_dataset_chunk['y'],
                                                                 batch_size=batch_size,
                                                                 shuffle=shuffle)
        return _batch_generators_dict

    def read_hd5_files(self):

        _empty_dict = {'x': [], 'y': []}
        _dataset = {'train': copy.deepcopy(_empty_dict),
                    'validation': copy.deepcopy(_empty_dict),
                    'test': copy.deepcopy(_empty_dict)}
        dataset_keys = ['train', 'validation', 'test']
        for chunk in tqdm(dataset_keys):
            with h5py.File(self.hd5_files_path[chunk], 'r') as file:
                # print('Reading hd5 from file = {0}'.format(file.file))
                _dataset[chunk]['x'] = file['x'][()]  # The weird sign at the end get all the values from the dataset
                _dataset[chunk]['x'] = file['x'][()]  # The weird sign at the end get all the values from the dataset
                _dataset[chunk]['y'] = file['y'][()]  # The weird sign at the end get all the values from the dataset
        logging.info('Dataset Loaded from h5py file')
        return _dataset

    def save_hd5_files(self, dataset_dict):
        if self.data_folder_path is not None:
            logging.info('Dataset saving to h5py file')
            for chunk in tqdm(dataset_dict):
                with h5py.File(self.hd5_files_path[chunk], 'w') as file:
                    # temp = np.array()
                    _ = file.create_dataset('x', data=dataset_dict[chunk]['x'])
                    _ = file.create_dataset('y', data=dataset_dict[chunk]['y'])

    def check_hd5_files(self):
        return os.path.isfile(self.hd5_files_path['train']) and \
               os.path.isfile(self.hd5_files_path['validation']) and \
               os.path.isfile(self.hd5_files_path['test'])

    @staticmethod
    def can_load_data_into_memory(dataset_size_in_gbs=None):
        if dataset_size_in_gbs is None:
            dataset_size_in_gbs = 110  # Debug value
        total_memory_in_gbs = psutil.virtual_memory().total / (1024 * 1024 * 1024)
        return total_memory_in_gbs > dataset_size_in_gbs

    def batch_generator(self, data_x, data_y, shuffle=True, batch_size=None):
        n = len(data_x)
        # data_y1 = data_y[:,0]
        # data_y2 = data_y[:,1]
        data_y1 = data_y

        if batch_size is None:
            batch_size = self.batch_size
        while True:
            batch_start = 0
            batch_end = batch_size
            indexes = np.arange(len(range(0, n)))
            if shuffle:
                np.random.shuffle(indexes)
            while batch_start < n:
                limit = min(batch_end, n)
                index = indexes[batch_start:limit]
                # for multiple labels
                # batch_y1 = [ data_y1[i] for i in index]
                # batch_y2 = [ data_y2[i] for i in index]
                # for single label
                batch_y1 = [data_y1[i] for i in index]
                batch_x = [data_x[i] for i in index]

                # try:
                batch_images_x = np.array(
                    [img_to_array(
                        (load_img(os.path.join(self.base_path, image_name)).resize(self.image_shape, Image.ANTIALIAS)))
                        for
                        image_name in batch_x])
                batch_images_x /= 255
                # print(batch_images_x)
                batch_images_y1 = np.array([y for y in batch_y1])

                # batch_imagesY2  =np.array([y for y in batch_y2])
                # yield(batch_images_x, [batch_images_y1, batch_imagesY2]) #for multiple outputs
                yield (batch_images_x, batch_images_y1)  # for single output
                batch_start += batch_size
                batch_end += batch_size
