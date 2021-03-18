"""A script providing inputs to the system.
This module contains a Class handling the inputs into the system
"""
from src.input_feed.image_feed import ImageFeed, camera_feed, video_feed, folder_feed
from src.input_feed.location_feed import LocationFeed, position_feed
from src.main.flags_global import FLAGS


class InputFeed:
    def __init__(self):
        self.data_batch = {}
        self.data_batch_ready = False
        self.data_sources_assigned = False

        self.Data = {'ImageData': [], 'LocationData': None}
        self.DataSources = {'ImageSource': ImageFeed.ImageFeedEmpty(),
                            'LocationSource': LocationFeed.LocationFeedEmpty()}
        self.__assign_sources()

    # noinspection PyTypeChecker
    def __assign_sources(self):
        # Assigning Image Sources
        if FLAGS.image_input_mode == 'camera':
            self.DataSources['ImageSource'] = camera_feed.CameraFeedAsync()
        elif FLAGS.image_input_mode == 'video':
            self.DataSources['ImageSource'] = video_feed.VideoFeedAsync()
        elif FLAGS.image_input_mode == 'folder':
            self.DataSources['ImageSource'] = folder_feed.FolderFeedSync()
        else:
            self.DataSources['ImageSource'] = None
        # Assigning Location Sources
        if FLAGS.location_input_mode == 'pos_constant':
            self.DataSources['LocationSource'] = position_feed.PosConstFeedSync()
        else:
            self.DataSources['LocationSource'] = None

        for datasource_name, datasource in self.DataSources.items():
            if datasource is None:
                raise ValueError(f"{datasource_name} has not been assigned. Check FLAGS.")

        self.data_sources_assigned = True

    def get_next_input_data(self):
        if self.data_sources_assigned is False:
            raise AttributeError("DataSources have not been assigned")
        self.Data['ImageData'] = self.DataSources['ImageSource'].get_frame()
        self.Data['LocationData'] = self.DataSources['LocationSource'].get_location()
        self.validate_input_data(ignore_validation=True)  # TODO: DEBUG ONLY !

    def validate_input_data(self, ignore_validation=False):
        if ignore_validation is True:
            return True
        for key, value in self.Data.items():
            if value is None:
                raise ValueError(f"{key} next value is None. Input feed interrupted. ")
            # TODO: more checks
        return True
        # TODO: raise errors if some data is invalid

    @staticmethod
    def validate_data_batch(data_batch):
        for key, value in data_batch:
            if value is None:
                return False
        return True

    def next_input_data_batch(self):
        self.get_next_input_data()
        self.data_batch = self.Data.copy()
        self.data_batch_ready = True

    def get_input_data_batch(self):
        self.next_input_data_batch()
        if self.data_batch_ready is True:
            data_batch = self.data_batch.copy()
            data_batch_is_valid = self.validate_data_batch(data_batch)
            self.data_batch = {}
            self.data_batch_ready = False
            return data_batch, data_batch_is_valid
        else:
            raise AttributeError('Trying to get old values, create new input data batch.')


