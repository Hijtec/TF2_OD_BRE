"""A script providing inputs to the system.
This module contains a Class handling the inputs into the system
"""
from src.main.flags_global import FLAGS
from src.input_feed.image_feed import camera_feed, video_feed, folder_feed
from src.input_feed.location_feed import position_feed


class InputFeed:
    def __init__(self):
        self.data_batch = {}
        self.data_batch_ready = False
        self.data_sources_assigned = False

        self.Data = {'ImageData': [], 'LocationData': None}
        self.DataSources = {'ImageSource': None, 'LocationSource': None}
        self.__assign_sources()

    def __assign_sources(self):
        # Assigning Image Sources
        self.DataSources['ImageSource'] = camera_feed.CameraFeedAsync() if FLAGS.image_input_mode == 'camera' else None
        self.DataSources['ImageSource'] = video_feed.VideoFeedAsync() if FLAGS.image_input_mode == 'video' else None
        self.DataSources['ImageSource'] = folder_feed.FolderFeedSync() if FLAGS.image_input_mode == 'folder' else None
        # Assigning Location Sources
        self.DataSources['LocationSource'] = position_feed.PosConstFeedSync() \
            if FLAGS.location_input_mode == 'pos_constant' else None

        for datasource_name, datasource in self.DataSources:
            if datasource is None:
                raise ValueError(f"{datasource_name} has not been assigned. Check FLAGS.")

        self.data_sources_assigned = True

    def get_next_input_data(self):
        if self.data_sources_assigned is False:
            raise AttributeError("DataSources have not been assigned")
        self.Data['ImageData'] = self.DataSources['ImageSource'].get_next_frame()
        self.Data['LocationData'] = self.DataSources['LocationSource'].get_next_location()
        self.validate_input_data()

    def validate_input_data(self):
        for key, value in self.Data.items():
            if value is None:
                raise ValueError(f"{key} next value is None. Input feed interrupted. ")
            # TODO: more checks
        return True
        # TODO: raise errors if some data is invalid

    def create_input_data_batch(self):
        self.data_batch = self.Data
        self.data_batch_ready = True

    def receive_data_batch(self):
        self.data_batch = {}
        self.data_batch_ready = False
