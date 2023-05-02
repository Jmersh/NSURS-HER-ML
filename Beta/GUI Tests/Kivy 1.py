import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import time

from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.tabbedpanel import TabbedPanelItem
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserListView
from kivy.properties import ObjectProperty, StringProperty
from kivy.garden.matplotlib import FigureCanvasKivyAgg
import matplotlib.pyplot as plt

class ModelOptionsTab(TabbedPanelItem):
    model_options_spinner = ObjectProperty(None)
    train_button = ObjectProperty(None)
    save_button = ObjectProperty(None)
    model_label = ObjectProperty(None)
    time_label = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(ModelOptionsTab, self).__init__(**kwargs)
        self.model_options_spinner.values = ['Linear Regression', 'Random Forest']
        self.model = None

    def train_model(self, model_type):
        if model_type == 'Linear Regression':
            self.model = LinearRegression()
        elif model_type == 'Random Forest':
            # Implement Random Forest model
            pass
        else:
            # Implement other models
            pass

        # Load dataset
        dataset_path = self.parent.dataset_tab.selected_path
        dataset = pd.read_csv(dataset_path)

        # Split dataset into training and testing sets
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Train model and benchmark time
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()
        self.time_label.text = "Training time: {:.2f} seconds".format(end_time - start_time)

        # Show model name
        self.model_label.text = "Model: {}".format(model_type)

    def save_model(self):
        if self.model is None:
            return

        model_path = self.parent.dataset_tab.selected_path + "_model.pkl"
        joblib.dump(self.model, model_path)

class DatasetTab(TabbedPanelItem):
    file_chooser = ObjectProperty(None)
    selected_path = StringProperty('')

    def __init__(self, **kwargs):
        super(DatasetTab, self).__init__(**kwargs)
        self.file_chooser.path = ''

    def file_selected(self, selection):
        if len(selection) > 0:
            self.selected_path = selection[0]

class PlotPreviewTab(TabbedPanelItem):
    plot_box_layout = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(PlotPreviewTab, self).__init__(**kwargs)
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasKivyAgg(figure=self.fig)
        self.plot_box_layout.add_widget(self.canvas)

    def plot_data(self, X, y_pred):
        self.ax.cla()
        self.ax.scatter(X, y_pred, color='red')

