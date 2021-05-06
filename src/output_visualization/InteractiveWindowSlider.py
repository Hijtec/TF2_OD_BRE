"""A helper class providing a Matplotlib interactive window with slider for output visualisation.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, TextBox
import numpy as np


class InteractiveWindow:
    def __init__(self, list_of_images, list_of_descriptions_labels, list_of_descriptions, paths_given=False):
        self.images = []
        self.n_of_images = None
        self.list_of_descriptions_labels = list_of_descriptions_labels
        self.list_of_descriptions = list_of_descriptions

        self.figure = None
        self.axes = []
        self.image = None
        self.slider = None
        self.textbox = None

        self.setup_image_input(list_of_images, paths_given)
        self.setup_style()
        self.setup_figure_layout()
        self.setup_image(self.images[0])
        self.setup_slider()
        self.setup_textbox()
        plt.show()

    def setup_image_input(self, list_of_images, paths_given):
        if paths_given:
            for pth in list_of_images:
                self.images.append(plt.imread(pth))
        else:
            self.images = list_of_images
        self.n_of_images = int(len(self.images))

    @staticmethod
    def setup_style():
        mpl.rcdefaults()
        mpl.rcParams['axes.titlelocation'] = 'center'
        mpl.rcParams['axes.linewidth'] = 0.5
        mpl.rcParams['axes.titlesize'] = 'medium'
        mpl.rcParams['axes.labelsize'] = 'medium'
        mpl.rcParams['font.size'] = 4

    def setup_figure_layout(self):
        self.figure = plt.figure(dpi=400)
        gs = GridSpec(3, 3, hspace=0.8, width_ratios=[4, 1, 1], height_ratios=[1, 1, 5], wspace=0.2)
        self.axes.append(self.figure.add_subplot(gs[:-1, -1]))  # ax5
        self.axes.append(self.figure.add_subplot(gs[-1, -1]))  # ax6
        self.axes.append(self.figure.add_subplot(gs[:, :-1]))  # ax4

    def setup_image(self, image):
        self.image = self.axes[2].imshow(image)
        self.axes[2].set_title("Image")
        self.axes[2].set_axis_off()

    def setup_slider(self):
        self.axes[0].set_title("Výběr zobrazených prvků")
        self.slider = Slider(self.axes[0], "", 0, self.n_of_images, valinit=0, valstep=1,
                             orientation="horizontal", dragging=False)
        self.slider.on_changed(self.update)

    def setup_textbox(self, title="NoTitle", description="NoDescription"):
        self.axes[1].set_axis_off()
        self.axes[1].set_title(title)
        self.axes[1].text(0, 0.5, description, fontsize=3, ha='center', va='center', wrap=True)

    def update(self, val):
        value_read = int(self.slider.val)
        self.setup_image(self.images[value_read])
        self.setup_textbox(self.list_of_descriptions_labels[value_read], self.list_of_descriptions[value_read])
        self.figure.canvas.draw()
