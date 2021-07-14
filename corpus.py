
import os
# import tarfile
import urllib.request
import zipfile


class SpeechCorpusProvider:
    """
    Ensures the availability and downloads the speech corpus if necessary
    """

    BASE_URL = 'http://images.cocodataset.org/zips/'
    DATA_SETS = ['test2014']
    SET_FILE_EXTENSION = '.zip'
    TAR_ROOT = 'LibriSpeech/'

    def __init__(self, data_directory):
        """
        Creates a new SpeechCorpusProvider with the root directory `data_directory`.
        The speech corpus is downloaded and extracted into sub-directories.

        Args:
            data_directory: the root directory to use, e.g. data/
        """

        self._data_directory = data_directory
        self._make_dir_if_not_exists(data_directory)

    @staticmethod
    def _make_dir_if_not_exists(directory):
        """
        Helper function to create a directory if it doesn't exist.

        Args:
            directory: directory to create
        """

        if not os.path.exists(directory):
            os.makedirs(directory)

    def _download_if_not_exists(self, remote_file_name):
        """
        Downloads the given `remote_file_name` if not yet stored in the `data_directory`

        Args:
            remote_file_name: the file to download

        Returns: path to downloaded file
        """

        print(SpeechCorpusProvider.BASE_URL + remote_file_name)

        path = os.path.join(self._data_directory, remote_file_name)
        if not os.path.exists(path):
            print('Downloading {}...'.format(remote_file_name))
            urllib.request.urlretrieve(SpeechCorpusProvider.BASE_URL + remote_file_name, path)
        return path

    @staticmethod
    def _extract_from_to(zip_file_path, target_directory):
        """
        Extract all necessary files `source` from `tar_file_name` into `target_directory`

        Args:
            tar_file_name: the tar file to extract from
            source: the directory in the root to extract
            target_directory: the directory to store the files in
        """
        print('Extracting {}...'.format(zip_file_path))
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(target_directory)


    def _is_ready(self, data_sets=DATA_SETS):
        """
        Returns whether all `data_sets` are downloaded and extracted

        Args:
            data_sets: list of the datasets to ensure

        Returns: bool, is ready to use

        """

        return all([os.path.exists(os.path.join(self._data_directory, data_set)) 
            for data_set in data_sets])

    def _download(self, data_sets=DATA_SETS):
        """
        Download the given `data_sets`

        Args:
            data_sets: a list of the datasets to download
        """

        for data_set in data_sets:
            remote_file = data_set + SpeechCorpusProvider.SET_FILE_EXTENSION
            self._download_if_not_exists(remote_file)

    def _extract(self, data_sets=DATA_SETS):
        """
        Extract all necessary files from the given `data_sets`

        Args:
            data_sets: a list of the datasets to extract the files from
        """

        for data_set in data_sets:
            local_file = os.path.join(self._data_directory, data_set + SpeechCorpusProvider.SET_FILE_EXTENSION)
            target_directory = self._data_directory
            self._extract_from_to(local_file, target_directory)

    def ensure_availability(self):
        """
        Ensure that all datasets are downloaded and extracted. If this is not the case,
        the download and extraction is initated.

        Args:
            test_only: Whether to exclude training and development data
        """

        data_set = SpeechCorpusProvider.DATA_SETS

        if not self._is_ready(data_set):
            self._download(data_set)
            self._extract(data_set)


if __name__=="__main__":
    corpus = SpeechCorpusProvider("images")
    corpus.ensure_availability()