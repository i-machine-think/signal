import os

from enums.dataset_type import DatasetType

DATA_FOLDER = 'data'
CHECKPOINTS_FOLDER = 'checkpoints'
RUNS_FOLDER = 'runs'
FEATURES_FOLDER = 'features'

class FileHelper:
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self._data_path = os.path.join(os.path.dirname(dir_path), DATA_FOLDER)
        
        if not os.path.exists(self._data_path):
            os.mkdir(self._data_path)

        features_folder_path = os.path.join(self._data_path, FEATURES_FOLDER)
        if not os.path.exists(features_folder_path):
            os.mkdir(features_folder_path)
            
        checkpoints_folder_path = os.path.join(self._data_path, CHECKPOINTS_FOLDER)
        if not os.path.exists(checkpoints_folder_path):
            os.mkdir(checkpoints_folder_path)
            

        self._model_checkpoint_path = os.path.join(self._data_path, CHECKPOINTS_FOLDER, 'extractor.p')

        self._train_input_path = os.path.join(self._data_path, FEATURES_FOLDER, 'train.input.npy')
        self._valid_input_path = os.path.join(self._data_path, FEATURES_FOLDER, 'valid.input.npy')
        self._test_input_path = os.path.join(self._data_path, FEATURES_FOLDER, 'test.input.npy')

        self._train_features_path = os.path.join(self._data_path, FEATURES_FOLDER, 'train_features.npy')
        self._valid_features_path = os.path.join(self._data_path, FEATURES_FOLDER, 'valid_features.npy')
        self._test_features_path = os.path.join(self._data_path, FEATURES_FOLDER, 'test_features.npy')

        self._train_metadata_path = os.path.join(self._data_path, FEATURES_FOLDER, 'train.metadata.p')
        self._valid_metadata_path = os.path.join(self._data_path, FEATURES_FOLDER, 'valid.metadata.p')
        self._test_metadata_path = os.path.join(self._data_path, FEATURES_FOLDER, 'test.metadata.p')

    @property
    def model_checkpoint_path(self):
        return self._model_checkpoint_path

    @property
    def train_input_path(self):
        return self._train_input_path

    @property
    def valid_input_path(self):
        return self._valid_input_path

    @property
    def test_input_path(self):
        return self._test_input_path

    @property
    def train_features_path(self):
        return self._train_features_path

    @property
    def valid_features_path(self):
        return self._valid_features_path

    @property
    def test_features_path(self):
        return self._test_features_path

    @property
    def train_metadata_path(self):
        return self._train_metadata_path

    @property
    def valid_metadata_path(self):
        return self._valid_metadata_path

    @property
    def test_metadata_path(self):
        return self._test_metadata_path

    def get_run_folder(self, sub_folder, model_name):
        if not sub_folder:
            run_folder = os.path.join(RUNS_FOLDER, model_name)
        else:
            run_folder = os.path.join(RUNS_FOLDER, sub_folder, model_name)

        return run_folder

    def get_sender_path(self, run_folder):
        result = os.path.join(run_folder, 'sender.p')
        return result
        
    def get_receiver_path(self, run_folder):
        result = os.path.join(run_folder, 'receiver.p')
        return result

    def get_input_path(self, dataset_type: DatasetType):
        if dataset_type == DatasetType.Train:
            return self.train_input_path
        elif dataset_type == DatasetType.Valid:
            return self.valid_input_path
        elif dataset_type == DatasetType.Test:
            return self.test_input_path
    
    def get_features_path(self, dataset_type: DatasetType):
        if dataset_type == DatasetType.Train:
            return self.train_features_path
        elif dataset_type == DatasetType.Valid:
            return self.valid_features_path
        elif dataset_type == DatasetType.Test:
            return self.test_features_path
    
    def get_metadata_path(self, dataset_type: DatasetType):
        if dataset_type == DatasetType.Train:
            return self.train_metadata_path
        elif dataset_type == DatasetType.Valid:
            return self.valid_metadata_path
        elif dataset_type == DatasetType.Test:
            return self.test_metadata_path

    def get_vocabulary_path(self, vocabulary_size: int):
        vocabulary_path = os.path.join(self._data_path, f'dict_{vocabulary_size}.pckl')
        return vocabulary_path

    def get_set_path(self, set_name: str):
        set_path = os.path.join(self._data_path, f'{set_name}.input')
        return set_path

    def create_unique_sender_path(self, model_name: str):
        sender_path = os.path.join(self._data_path, CHECKPOINTS_FOLDER, f'sender_{model_name}.p')
        return sender_path