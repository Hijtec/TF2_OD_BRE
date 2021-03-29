"""A class distinguishing mutually exclusive elevator environments for output interpretation.
"""


class ElevatorEnvironmentDistinguisher:
    def __init__(self, max_floor_buttons_count=25):
        self.max_floor_buttons_count = max_floor_buttons_count
        self.location_environment = None
        self.floor_buttons_count = 0
        self.displays_count = 0
        self.call_panels_count = 0
        self.auxiliary_buttons_count = 0
        self.communication_buttons_count = 0

        self.distinguished_elevator_environment = None

    def _count_class_instances_from_detection(self):
        pass  # TODO: implement

    def is_elevator_panel(self):
        name = "ElevatorPanel"
        reqs_met = self._meet_requirements(["CabinPanel"], flr_min=2, flr_max=self.max_floor_buttons_count,
                                           aux_min=1, com_min=1, dsp_max=1)
        return reqs_met, name

    def is_floor_display(self):
        name = "FloorDisplay"
        reqs_met = self._meet_requirements(["CabinPanel", "Cabin", "FrontOfElevator"], flr_max=1, flr_min=1,
                                           aux_max=1, com_max=1)
        return reqs_met, name

    def is_call_elevator(self):
        name = "CallElevator"
        reqs_met = False
        if self.call_panels_count >= 1:
            reqs_met = self._meet_requirements(["FrontOfElevator"], flr_min=0, flr_max=4, dsp_max=1)
            return reqs_met, name
        else:
            return reqs_met, name

    @staticmethod
    def is_null_env_default():
        name = "Other"
        reqs_met = True
        return reqs_met, name

    def _meet_requirements(self, location_required_name=None,
                           flr_min=0, flr_max=50, aux_min=0, aux_max=10, com_min=0,
                           com_max=10, dsp_min=0, dsp_max=10):
        if (flr_min <= self.floor_buttons_count <= flr_max and
                aux_min <= self.auxiliary_buttons_count <= aux_max and
                com_min <= self.communication_buttons_count <= com_max and
                dsp_min <= self.displays_count <= dsp_max and
                self.location_environment in location_required_name):
            return True
        else:
            return False

    def distinguish_elevator_environment(self):
        if self.is_elevator_panel()[0]:
            self.distinguished_elevator_environment = self.is_elevator_panel()[1]
        elif self.is_floor_display()[0]:
            self.distinguished_elevator_environment = self.is_floor_display()[1]
        elif self.is_call_elevator()[0]:
            self.distinguished_elevator_environment = self.is_call_elevator()[1]
        else:
            self.distinguished_elevator_environment = self.is_null_env_default()[1]

