"""A postprocessing module for button recognition.

#Input:         Raw detections of buttons(array), button width and height(floats)
#Output:        Proposed corrected array of elevator panel(numpy.array)
#Assumptions:   TODO:All numbered buttons were detected(array/list of lists), localized(float) and labeled(int)
                First and last buttons are correctly labeled
                Buttons detection data is ordered from bottom to top (y_min -> y_max) and from left to right

This script utilises data given from a recognition module and feeds it
for postprocessing in order to correct and enhance the data for further
use. These include:
    assigning columns and rows
    creating callable objects
    finding correct button panel template
    fix number sequence

  Typical usage example:

  det = Detection(detection_data,softmax_prediction,button_width,button_height)
"""

from math import floor as rounddown
from src.utils.relabeling_utils import Button, Panel, Template
from absl import logging
import numpy as np


class PrepareRelabelingInput:
    """Class to retrofit neural network classifications to match the RelabelButtons class."""
    # TODO: implement


class RelabelButtons:
    """A wrapper class for data input.

    Attributes:
        detected:                   An array of given data from recognition of type [[x, y, num_det]..]
        buttons_raw:                List of Button classes based on given data
        template:                   Template object, corrects button sequence
        panel:                      Panel object, holds information after the processed detection
        but:                        Tuple of detected button parameters (width, height)
        adj_coefficient: TODO:      A coefficient for moving the comparison value for imperfect/real positions
        softmax_pred:               A list of lists of x',x'',x''' prediction of label of buttons from given data

    Methods:
        create_buttons_raw:         Creates a list of Button objects from raw data
        create_panel:               Creates a slave Panel object
        create_template:            Creates a slave Template object
        find_classes(axis = 1/0):   Finds buttons along the same row/column
        order_unique_coord(coord, history, type = "rows"/"cols"):
                                    Rearranges rows and columns based on their average position in space
    """
    def __init__(self, detected_buttons_coord, softmax_pred, but_width, but_height):
        """Initializes the class and calls its methods."""
        self.detected = detected_buttons_coord
        self.softmax_pred = softmax_pred
        self.but = (but_width, but_height)

        # Objects used
        self.buttons_raw = None
        self.buttons = None
        self.template = None
        self.panel = None

        # Parameters
        self.adj_coefficient = 1

        # Methods called
        self.check_given_inputs()
        self.create_buttons_raw()
        self.create_template()
        self.create_panel()

    def check_given_inputs(self):
        for i in self.detected:
            if len(i) != 3:
                logging.fatal('The detected button input is of wrong type')

    def create_buttons_raw(self):
        """Creates a list of button objects based on raw data."""
        button_list = []
        for i in self.detected:
            x_raw, y_raw, n_raw = i[0], i[1], i[2]
            button_list.append(Button(x_raw, y_raw, n_raw))
        self.buttons_raw = button_list

    def create_template(self):
        """Creates a template object."""
        rows, rows_all, r_val_hist = self.find_classes("row")
        cols, cols_all, c_val_hist = self.find_classes("col")

        if len(rows) > 1/self.but[1]:
            logging.fatal("There can't be more rows than can physically fit into state space!")
        if len(cols) > 1/self.but[0]:
            logging.fatal("There can't be more cols than can physically fit into state space!")

        rows_ordered = self.order_unique_coord(rows_all, r_val_hist, "row")
        cols_ordered = self.order_unique_coord(cols_all, c_val_hist, "col")

        self.template = Template(self.buttons_raw, self.softmax_pred, rows_ordered, cols_ordered)

    def create_panel(self):
        """Creates a panel object using passing of created Template attributes."""
        self.panel = Panel(self.template.buttons_ordered_corrected,
                           self.template.rows,
                           self.template.cols,
                           self.template.priority_lr,
                           self.template.priority_vh)

    def find_classes(self, axis=0):
        """Finds rows/columns in data.

        Args:
            axis:                   Integer, 0 for column, 1 for row

        Returns:
            sames_unique:           Numpy sorted array of unique classes
            sames:                  Numpy array of found classes
            comp_val_history:       List of average values of y_raw (for rows)/ x_raw(for cols)
                                    for each member, used to differentiate classes
        """
        i = -1
        sames = []
        sames_unique = []
        comp_val_history = []

        for _ in self.detected:
            same_class = []
            j = -1

            i += 1
            compare_val = self.detected[i][axis]

            for d in self.detected:
                j += 1
                val = d[axis]
                if abs(val - compare_val) < self.but[axis] / 2:
                    same_class.append(j)
                    compare_val = compare_val + (val - compare_val) / self.adj_cooef

            sames.append(same_class)
            comp_val_history.append(
                rounddown(compare_val * 10)
            )  # Important to sorting the columns based on y_axis

        sames = np.array(sames)
        sames_unique = np.unique(sames)

        return sames_unique, sames, comp_val_history

    @staticmethod
    def order_unique_coord(coord, comp_hist, axis):
        """Rearranges rows/cols based on their position and eliminates duplicities.

        Args:
            coord:      List of lists of all detected rows/columns
            comp_hist:  List of average values of y_raw (for rows)/ x_raw(for cols)
                        for each member, used to differentiate classes
            axis:       str("row"/"col") based on finding order of rows/columns

        Returns:
            out:        Returns the ordered list of rows/cols

        Raises:
            TypeError:  When arg type is not either "col" or "row"
        """
        rearranged, out = [], []
        indexing = np.argsort(comp_hist)

        for j in indexing:
            rearranged.append(coord[j])
        _, idx = np.unique(rearranged, return_index=True)

        for j in np.sort(idx):
            out.append(rearranged[j])

        return out
