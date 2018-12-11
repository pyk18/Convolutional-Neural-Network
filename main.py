# Arora, Priyank
# 1001-55-3349
# 2018-12-09
# Assignment-06-01
import sys
import sklearn.datasets
from sklearn.cluster import AgglomerativeClustering

import pickle
import os
from keras.models import load_model
from keras.models import Sequential
from keras import regularizers
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense

if sys.version_info[0] < 3:
	import Tkinter as tk
else:
	import tkinter as tk
from tkinter import simpledialog
from tkinter import filedialog
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from PIL import Image
from os import listdir
from os.path import isfile, join

import math
import tensorflow as tf
import random

from keras.utils import to_categorical

import matplotlib.backends.tkagg as tkagg
np.seterr(all="raise")

class MainWindow(tk.Tk):
	"""
	This class creates and controls the main window frames and widgets
	
	"""

	def __init__(self, debug_print_flag=False):
		tk.Tk.__init__(self)
		self.debug_print_flag = debug_print_flag
		self.master_frame = tk.Frame(self)
		self.master_frame.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
		self.rowconfigure(0, weight=1, minsize=500)
		self.columnconfigure(0, weight=1, minsize=500)
		
		self.master_frame.rowconfigure(2, weight=10, minsize=400, uniform='xx')
		self.master_frame.rowconfigure(3, weight=1, minsize=10, uniform='xx')
		self.master_frame.columnconfigure(0, weight=1, minsize=200, uniform='xx')
		self.master_frame.columnconfigure(1, weight=1, minsize=200, uniform='xx')
		# create all the widgets
		self.menu_bar = MenuBar(self, self.master_frame, background='orange')
		self.tool_bar = ToolBar(self, self.master_frame)
		self.left_frame = tk.Frame(self.master_frame)
		#RF:
		#self.right_frame = tk.Frame(self.master_frame)
		self.status_bar = StatusBar(self, self.master_frame, bd=1, relief=tk.SUNKEN)
		# Arrange the widgets
		self.menu_bar.grid(row=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
		self.tool_bar.grid(row=1, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
		self.left_frame.grid(row=2, column=0, columnspan=4, sticky=tk.N + tk.E + tk.S + tk.W)
		
		self.status_bar.grid(row=3, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
		
		self.display_activation_functions = LeftFrame(self, self.left_frame, debug_print_flag=self.debug_print_flag)
	

class MenuBar(tk.Frame):
	def __init__(self, root, master, *args, **kwargs):
		tk.Frame.__init__(self, master, *args, **kwargs)
		self.root = root
		self.menu = tk.Menu(self.root)
		root.config(menu=self.menu)
		self.file_menu = tk.Menu(self.menu)
		self.menu.add_cascade(label="File", menu=self.file_menu)
		self.file_menu.add_command(label="New", command=self.menu_callback)
		self.file_menu.add_command(label="Open...", command=self.menu_callback)
		self.file_menu.add_separator()
		self.file_menu.add_command(label="Exit", command=self.close_program)
		self.dummy_menu = tk.Menu(self.menu)
		self.menu.add_cascade(label="Dummy", menu=self.dummy_menu)
		self.dummy_menu.add_command(label="Item1", command=self.menu_item1_callback)
		self.dummy_menu.add_command(label="Item2", command=self.menu_item2_callback)
		self.help_menu = tk.Menu(self.menu)
		self.menu.add_cascade(label="Help", menu=self.help_menu)
		self.help_menu.add_command(label="About...", command=self.menu_help_callback)

	def menu_callback(self):
		self.root.status_bar.set('%s', "called the menu callback!")

	def close_program(self):
		self.root.destroy()
		self.root.quit()
		
	def menu_help_callback(self):
		self.root.status_bar.set('%s', "called the help menu callback!")

	def menu_item1_callback(self):
		self.root.status_bar.set('%s', "called item1 callback!")

	def menu_item2_callback(self):
		self.root.status_bar.set('%s', "called item2 callback!")


class ToolBar(tk.Frame):
	def __init__(self, root, master, *args, **kwargs):
		tk.Frame.__init__(self, master, *args, **kwargs)
		self.root = root
		self.master = master
		self.var_filename = tk.StringVar()
		self.var_filename.set('')
		self.ask_for_string = tk.Button(self, text="Ask for a string", command=self.ask_for_string)
		self.ask_for_string.grid(row=0, column=1)
		self.file_dialog_button = tk.Button(self, text="Open File Dialog", fg="blue", command=self.browse_file)
		self.file_dialog_button.grid(row=0, column=2)
		self.open_dialog_button = tk.Button(self, text="Open Dialog", fg="blue", command=self.open_dialog_callback)
		self.open_dialog_button.grid(row=0, column=3)

	def say_hi(self):
		self.root.status_bar.set('%s', "hi there, everyone!")

	def ask_for_string(self):
		s = simpledialog.askstring('My Dialog', 'Please enter a string')
		self.root.status_bar.set('%s', s)

	def ask_for_float(self):
		f = float(simpledialog.askfloat('My Dialog', 'Please enter a float'))
		self.root.status_bar.set('%s', str(f))

	def browse_file(self):
		self.var_filename.set(tk.filedialog.askopenfilename(filetypes=[("allfiles", "*"), ("pythonfiles", "*.txt")]))
		filename = self.var_filename.get()
		self.root.status_bar.set('%s', filename)

	def open_dialog_callback(self):
		d = MyDialog(self.root)
		self.root.status_bar.set('%s', "mydialog_callback pressed. Returned results: " + str(d.result))

	def button2_callback(self):
		self.root.status_bar.set('%s', 'button2 pressed.')

	def toolbar_draw_callback(self):
		self.root.display_graphics.create_graphic_objects()
		self.root.status_bar.set('%s', "called the draw callback!")

	def toolbar_callback(self):
		self.root.status_bar.set('%s', "called the toolbar callback!")


class MyDialog(tk.simpledialog.Dialog):
	def body(self, parent):
		tk.Label(parent, text="Integer:").grid(row=0, sticky=tk.W)
		tk.Label(parent, text="Float:").grid(row=1, column=0, sticky=tk.W)
		tk.Label(parent, text="String:").grid(row=1, column=2, sticky=tk.W)
		self.e1 = tk.Entry(parent)
		self.e1.insert(0, 0)
		self.e2 = tk.Entry(parent)
		self.e2.insert(0, 4.2)
		self.e3 = tk.Entry(parent)
		self.e3.insert(0, 'Default text')
		self.e1.grid(row=0, column=1)
		self.e2.grid(row=1, column=1)
		self.e3.grid(row=1, column=3)
		self.cb = tk.Checkbutton(parent, text="Hardcopy")
		self.cb.grid(row=3, columnspan=2, sticky=tk.W)

	def apply(self):
		try:
			first = int(self.e1.get())
			second = float(self.e2.get())
			third = self.e3.get()
			self.result = first, second, third
		except ValueError:
			tk.tkMessageBox.showwarning("Bad input", "Illegal values, please try again")


class StatusBar(tk.Frame):
	def __init__(self, root, master, *args, **kwargs):
		tk.Frame.__init__(self, master, *args, **kwargs)
		self.label = tk.Label(self)
		self.label.grid(row=0, sticky=tk.N + tk.E + tk.S + tk.W)

	def set(self, format, *args):
		self.label.config(text=format % args)
		self.label.update_idletasks()

	def clear(self):
		self.label.config(text="")
		self.label.update_idletasks()


class LeftFrame:
	"""
	This class creates and controls the widgets and figures in the left frame which
	are used to display the activation functions.
	
	"""

	def __init__(self, root, master, debug_print_flag=False):
		self.master = master
		self.root = root
		#########################################################################
		#  Initilizing the constants values
		#########################################################################
		self.xmin = 0
		self.xmax = 50
		self.ymin = 0
		self.ymax = 1.0
		self.alpha = 0.1
		self.lamda = 0.01
		self.F1 = 32
		self.K1 = 3
		self.F2 = 32
		self.K2 = 3
		self.F3 = 32
		self.K3 = 3
		self.samples = 20
		self.load_saved_model = False
		self.model_name = os.path.join('.', 'Data', 'cifar10_model.h5')
		self.gen_data()
		self.dataset_load()
		self.epoch_list = []
		
		#########################################################################
		#  Set up the plotting frame and controls frame
		#########################################################################
		master.rowconfigure(0, weight=10, minsize=200)
		master.columnconfigure(0, weight=1)
		self.plot_frame = tk.Frame(self.master, borderwidth=10, relief=tk.SUNKEN)
		self.plot_frame.grid(rowspan=4, columnspan=4, sticky=tk.N + tk.E + tk.S + tk.W)
		
		self.figure, (ax1, ax2) = plt.subplots(1,2)
		plt.subplots_adjust(left=0.1, bottom=0.4, right=0.9, top=0.9)
		self.axes = [ax1, ax2]
		self.epoch_list = []
		self.axes[0].set_xlabel('No. of Epochs')
		self.axes[0].set_ylabel('Error %')
		self.axes[0].set_autoscale_on(False)
		
		self.axes[0].set_xlim(self.xmin, self.xmax)
		self.axes[0].set_ylim(self.ymin, self.ymax)
		self.axes[1].set_xticks([])
		self.axes[1].set_yticks([])
		self.axes[1].set_title("Confusion Matrix")
		self.axes[1].set_xlabel("Actual Value")
		self.axes[1].set_ylabel("Predicted Value")
		self.epochs = 0
		self.epoch_list = []
		
		self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
		self.plot_widget = self.canvas.get_tk_widget()
		self.plot_widget.grid(row=0, columnspan=4, sticky=tk.N + tk.E + tk.S + tk.W)
		self.plot_widget.pack(fill="both", expand=True)
		
		self.controls_frame = tk.Frame(self.master)
		self.controls_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
		#########################################################################
		#  Set up the sliders and selection boxes
		#########################################################################
		self.alpha_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL, from_=0.000,
		                            to_=1.0, resolution=0.001, bg="#DDDDDD", activebackground="#FF0000",
		                            highlightcolor="#00FFFF", label="Alpha",
		                            command=lambda event: self.alpha_slider_callback())
		self.alpha_slider.set(self.alpha)
		self.alpha_slider.bind("<ButtonRelease-1>", lambda event: self.alpha_slider_callback())
		self.alpha_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
		
		
		self.lambda_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL, from_=0.000,
		                            to_=1.0, resolution=0.01, bg="#DDDDDD", activebackground="#FF0000",
		                            highlightcolor="#00FFFF", label="Lambda",
		                            command=lambda event: self.lambda_slider_callback())
		self.lambda_slider.set(self.lamda)
		self.lambda_slider.bind("<ButtonRelease-1>", lambda event: self.lambda_slider_callback())
		self.lambda_slider.grid(row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
		
		
		self.F1_slider = tk.Scale(self.controls_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL, from_=1,
		                            to_=64, resolution=1, bg="#DDDDDD", activebackground="#FF0000",
		                            highlightcolor="#00FFFF", label="F1",
		                            command=lambda event: self.F1_slider_callback())
		self.F1_slider.set(self.F1)
		self.F1_slider.bind("<ButtonRelease-1>", lambda event: self.F1_slider_callback())
		self.F1_slider.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
		
		self.K1_slider = tk.Scale(self.controls_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL, from_=3,
		                            to_=7, resolution=1, bg="#DDDDDD", activebackground="#FF0000",
		                            highlightcolor="#00FFFF", label="K1",
		                            command=lambda event: self.K1_slider_callback())
		self.K1_slider.set(self.K1)
		self.K1_slider.bind("<ButtonRelease-1>", lambda event: self.K1_slider_callback())
		self.K1_slider.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
		
		self.F2_slider = tk.Scale(self.controls_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL, from_=1,
		                            to_=64, resolution=1, bg="#DDDDDD", activebackground="#FF0000",
		                            highlightcolor="#00FFFF", label="F2",
		                            command=lambda event: self.F2_slider_callback())
		self.F2_slider.set(self.F2)
		self.F2_slider.bind("<ButtonRelease-1>", lambda event: self.F2_slider_callback())
		self.F2_slider.grid(row=1, column=3, sticky=tk.N + tk.E + tk.S + tk.W)
		
		self.K2_slider = tk.Scale(self.controls_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL, from_=3,
		                            to_=7, resolution=1, bg="#DDDDDD", activebackground="#FF0000",
		                            highlightcolor="#00FFFF", label="K2",
		                            command=lambda event: self.K2_slider_callback())
		self.K2_slider.set(self.K1)
		self.K2_slider.bind("<ButtonRelease-1>", lambda event: self.K2_slider_callback())
		self.K2_slider.grid(row=1, column=4, sticky=tk.N + tk.E + tk.S + tk.W)
		
		
		self.samples_slider = tk.Scale(self.controls_frame, variable=tk.IntVar(), orient=tk.HORIZONTAL, from_=0,
		                            to_=100, resolution=1, bg="#DDDDDD", activebackground="#FF0000",
		                            highlightcolor="#00FFFF", label="No. of Samples",
		                            command=lambda event: self.samples_slider_callback())
		self.samples_slider.set(self.samples)
		self.samples_slider.bind("<ButtonRelease-1>", lambda event: self.samples_slider_callback())
		self.samples_slider.grid(row=0, column=3, sticky=tk.N + tk.E + tk.S + tk.W)
		
		self.train = tk.Button(self.controls_frame, text="Adjust Weights (Train)", fg="black", width=20,
		                             command=self.train_callback)
		self.train.grid(row=0, column=6)
		
		self.randomize_weights = tk.Button(self.controls_frame, text="Reset Weights", fg="black", width=20,
		                             command=self.randomize_weights_callback)
		self.randomize_weights.grid(row=0, column=7)
		self.randomize_weights_callback()
		#########################################################################
		#  Set up the frame for drop down selection
		#########################################################################
		self.canvas.get_tk_widget().bind("<ButtonPress-1>", self.left_mouse_click_callback)
		self.canvas.get_tk_widget().bind("<ButtonPress-1>", self.left_mouse_click_callback)
		self.canvas.get_tk_widget().bind("<ButtonRelease-1>", self.left_mouse_release_callback)
		self.canvas.get_tk_widget().bind("<B1-Motion>", self.left_mouse_down_motion_callback)
		self.canvas.get_tk_widget().bind("<ButtonPress-3>", self.right_mouse_click_callback)
		self.canvas.get_tk_widget().bind("<ButtonRelease-3>", self.right_mouse_release_callback)
		self.canvas.get_tk_widget().bind("<B3-Motion>", self.right_mouse_down_motion_callback)
		self.canvas.get_tk_widget().bind("<Key>", self.key_pressed_callback)
		self.canvas.get_tk_widget().bind("<Up>", self.up_arrow_pressed_callback)
		self.canvas.get_tk_widget().bind("<Down>", self.down_arrow_pressed_callback)
		self.canvas.get_tk_widget().bind("<Right>", self.right_arrow_pressed_callback)
		self.canvas.get_tk_widget().bind("<Left>", self.left_arrow_pressed_callback)
		self.canvas.get_tk_widget().bind("<Shift-Up>", self.shift_up_arrow_pressed_callback)
		self.canvas.get_tk_widget().bind("<Shift-Down>", self.shift_down_arrow_pressed_callback)
		self.canvas.get_tk_widget().bind("<Shift-Right>", self.shift_right_arrow_pressed_callback)
		self.canvas.get_tk_widget().bind("<Shift-Left>", self.shift_left_arrow_pressed_callback)
		self.canvas.get_tk_widget().bind("f", self.f_key_pressed_callback)
		self.canvas.get_tk_widget().bind("b", self.b_key_pressed_callback)
		
	def display_numpy_array_as_table(self, input_array):
		
		if input_array.ndim==1:
			num_of_columns,=input_array.shape
			temp_matrix=input_array.reshape((1, num_of_columns))
		elif input_array.ndim>2:
			
			return
		else:
			temp_matrix=input_array
		number_of_rows,num_of_columns = temp_matrix.shape
		tb = self.axes[1].table(cellText=np.round(temp_matrix,2), loc=(0,0), cellLoc='center')
		for cell in tb.properties()['child_artists']:
			cell.set_height(1/number_of_rows)
			cell.set_width(1/num_of_columns)
		self.axes[1].set_xticks([])
		self.axes[1].set_yticks([])
		self.canvas.draw()
	
	def unpickle(self, file):
		with open(file, 'rb') as fo:
			dict = pickle.load(fo, encoding='bytes')
		return dict
		
	def display_activation_function(self):
		import copy
		epoch_list = copy.copy(self.epoch_list)
		if len(epoch_list) > 0:
			epoch_list.insert(0, epoch_list[0])
		self.axes[0].plot(list(range(len(epoch_list))), epoch_list)
		self.axes[0].set_title("Error vs Epochs")
		self.axes[0].xaxis.set_visible(True)
		self.axes[0].yaxis.set_visible(True)
		#self.axes[0].set_xlim(self.xmin, self.xmax)
		self.axes[0].set_ylim(self.ymin, self.ymax)
		self.axes[0].set_xlabel('Epochs')
		self.axes[0].set_ylabel('Error in percent')
		self.canvas.draw()
		
	def gen_data(self):
		folder_name = 'Data'
		train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
		train_x, train_y_labels, test_x, test_y_labels = None, None, None, None
		for tr_file in train_files:
			data = self.unpickle(os.path.join('.',folder_name,tr_file))
			train = data.get(b'data').reshape(10000, 32, 32, 3)
			labels = data.get(b'labels')
			if train_x is None:
				train_x = train
				train_y_labels = np.array(labels)
			else:
				train_x = np.append(train_x, train, axis=0)
				train_y_labels = np.append(train_y_labels, np.array(labels))
		data = self.unpickle(os.path.join('.', folder_name, 'test_batch'))
		test_x = data.get(b'data').reshape(10000, 32, 32, 3)
		test_y_labels = np.array(data.get(b'labels'))
		train_x = train_x.astype('float32') / 255.
		test_x = test_x.astype('float32') / 255.
		train_y = to_categorical(train_y_labels)
		test_y = to_categorical(test_y_labels)
		self.train_x = train_x
		self.train_y = train_y
		self.train_y_labels = train_y_labels
		self.test_x = test_x
		self.test_y = test_y
		self.test_y_labels = test_y_labels
	
	def dataset_load(self):
		samples = self.samples / 100.
		total = math.ceil(samples * self.train_x.shape[0])
		self.t_x = self.train_x[:total, :, :, :]
		self.t_y = self.train_y[:total]
		self.t_y_labels = self.train_y_labels[:total]
	
	def K2_slider_callback(self):
		self.K2 = int(self.K2_slider.get())
		self.load_saved_model = False
		self.reset_error_chart()
	
	def F2_slider_callback(self):
		self.F2 = int(self.F2_slider.get())
		self.load_saved_model = False
		self.reset_error_chart()
		
	def K1_slider_callback(self):
		self.K1 = int(self.K1_slider.get())
		self.load_saved_model = False
		self.reset_error_chart()
		
	def F1_slider_callback(self):
		self.F1 = int(self.F1_slider.get())
		self.load_saved_model = False
		self.reset_error_chart()
	
	def generate_new_model(self):
		model = Sequential()

		model.add(Conv2D(filters=self.F1, 
						kernel_size=(self.K1, self.K1),
						activation='relu',
						strides=(1, 1),
						padding='same',
						input_shape=(32, 32, 3),
						bias_regularizer=regularizers.l2(self.lamda),
						kernel_regularizer=regularizers.l2(self.lamda)))
		model.add(MaxPool2D())

		model.add(Conv2D(filters=self.F2, 
						kernel_size=(self.K2, self.K2),
						strides=(1, 1),
						padding='same',
						activation='relu',
						bias_regularizer=regularizers.l2(self.lamda),
						kernel_regularizer=regularizers.l2(self.lamda)))
		model.add(MaxPool2D())

		model.add(Conv2D(filters=self.F3, 
						kernel_size=(self.K3, self.K3),
						strides=(1, 1),
						padding='same',
						activation='relu',
						bias_regularizer=regularizers.l2(self.lamda),
						kernel_regularizer=regularizers.l2(self.lamda)))
		model.add(MaxPool2D())

		model.add(Flatten())
		model.add(Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(self.lamda)))

		model.compile(optimizer='adam',
					 loss='categorical_crossentropy',
					 metrics=['accuracy'])
		return model
	
	def train_callback(self):
		if self.load_saved_model:
			model = load_model(self.model_name)
		else:
			model = self.generate_new_model()
		
		history = model.fit(self.t_x, self.t_y, batch_size=50, epochs=1, verbose=0, validation_data=(self.test_x, self.test_y))
		error = 1 - history.history['val_acc'][0]
		self.epoch_list.append(error)
		if len(self.epoch_list) > self.xmax:
			self.epoch_list = self.epoch_list[-self.xmax:]
		self.display_activation_function()
		model.save(self.model_name)
		self.load_saved_model = True
		y_predicted = model.predict(self.test_x)
		y_pred_classes = np.argmax(y_predicted, axis=1)
		y_actual_classes = self.test_y_labels
		cm = np.zeros((y_predicted.shape[1], y_predicted.shape[1]))
		for i in range(len(y_actual_classes)):
			cm[y_pred_classes[i]][y_actual_classes[i]] += 1
		row_sum = np.sum(cm, axis=1)
		row_sum[row_sum == 0] = 1
		for row_i in range(len(row_sum)):
			cm[row_i, :] /= row_sum[row_i]
		cm_percent = cm
		
		self.display_numpy_array_as_table(cm_percent)
	
	def key_pressed_callback(self, event):
		self.root.status_bar.set('%s', 'Key pressed')

	def up_arrow_pressed_callback(self, event):
		self.root.status_bar.set('%s', "Up arrow was pressed")

	def down_arrow_pressed_callback(self, event):
		self.root.status_bar.set('%s', "Down arrow was pressed")

	def right_arrow_pressed_callback(self, event):
		self.root.status_bar.set('%s', "Right arrow was pressed")

	def left_arrow_pressed_callback(self, event):
		self.root.status_bar.set('%s', "Left arrow was pressed")

	def shift_up_arrow_pressed_callback(self, event):
		self.root.status_bar.set('%s', "Shift up arrow was pressed")

	def shift_down_arrow_pressed_callback(self, event):
		self.root.status_bar.set('%s', "Shift down arrow was pressed")

	def shift_right_arrow_pressed_callback(self, event):
		self.root.status_bar.set('%s', "Shift right arrow was pressed")

	def shift_left_arrow_pressed_callback(self, event):
		self.root.status_bar.set('%s', "Shift left arrow was pressed")

	def f_key_pressed_callback(self, event):
		self.root.status_bar.set('%s', "f key was pressed")

	def b_key_pressed_callback(self, event):
		self.root.status_bar.set('%s', "b key was pressed")

	def left_mouse_click_callback(self, event):
		self.root.status_bar.set('%s', 'Left mouse button was clicked. ' + 'x=' + str(event.x) + '   y=' + str(
			event.y))
		self.x = event.x
		self.y = event.y
		self.canvas.focus_set()

	def left_mouse_release_callback(self, event):
		self.root.status_bar.set('%s',
		                         'Left mouse button was released. ' + 'x=' + str(event.x) + '   y=' + str(event.y))
		self.x = None
		self.y = None

	def left_mouse_down_motion_callback(self, event):
		self.root.status_bar.set('%s', 'Left mouse down motion. ' + 'x=' + str(event.x) + '   y=' + str(event.y))
		self.x = event.x
		self.y = event.y

	def right_mouse_click_callback(self, event):
		self.root.status_bar.set('%s', 'Right mouse down motion. ' + 'x=' + str(event.x) + '   y=' + str(event.y))
		self.x = event.x
		self.y = event.y

	def right_mouse_release_callback(self, event):
		self.root.status_bar.set('%s',
		                         'Right mouse button was released. ' + 'x=' + str(event.x) + '   y=' + str(event.y))
		self.x = None
		self.y = None

	def right_mouse_down_motion_callback(self, event):
		self.root.status_bar.set('%s', 'Right mouse down motion. ' + 'x=' + str(event.x) + '   y=' + str(event.y))
		self.x = event.x
		self.y = event.y

	def left_mouse_click_callback(self, event):
		self.root.status_bar.set('%s', 'Left mouse button was clicked. ' + 'x=' + str(event.x) + '   y=' + str(
			event.y))
		self.x = event.x
		self.y = event.y

	def alpha_slider_callback(self):
		self.alpha = float(self.alpha_slider.get())

	def lambda_slider_callback(self):
		self.lamda = float(self.lambda_slider.get())
		self.load_saved_model = False
		self.reset_error_chart()
	
	def samples_slider_callback(self):
		self.samples = int(self.samples_slider.get())
		self.dataset_load()
	
	def randomize_weights_callback(self):
		model = self.generate_new_model()
		
		model.save(self.model_name)
		self.load_saved_model = True
		y_predicted = model.predict(self.test_x)
		y_pred_classes = np.argmax(y_predicted, axis=1)
		y_actual_classes = self.test_y_labels
		cm = np.zeros((y_predicted.shape[1], y_predicted.shape[1]))
		for i in range(len(y_actual_classes)):
			cm[y_pred_classes[i]][y_actual_classes[i]] += 1
		row_sum = np.sum(cm, axis=1)
		row_sum[row_sum == 0] = 1
		for row_i in range(len(row_sum)):
			cm[row_i, :] /= row_sum[row_i]
		cm_percent = cm
		
		self.display_numpy_array_as_table(cm_percent)
		self.reset_error_chart()
	
	def reset_error_chart(self):
		self.epoch_list = []
		self.axes[0].cla()
		self.axes[0].clear()
		self.axes[0].relim()
		self.axes[0].autoscale_view()
		self.axes[0].set_xlim(self.xmin, self.xmax)
		self.axes[0].set_ylim(self.ymin, self.ymax)
		self.axes[0].set_xlabel('no. of Epochs')
		self.axes[0].set_ylabel('Error percentage')
		self.canvas.draw()

class RightFrame:
	"""
	This class is for creating right frame widgets which are used to draw graphics
	on canvas as well as embedding matplotlib figures in the tkinter.
	
	"""
	def __init__(self, root, master, debug_print_flag=False):
		self.root = root
		self.master = master
		self.debug_print_flag = debug_print_flag
		width_px = root.winfo_screenwidth()
		height_px = root.winfo_screenheight()
		width_mm = root.winfo_screenmmwidth()
		height_mm = root.winfo_screenmmheight()
		# 2.54 cm = in
		width_in = width_mm / 25.4
		height_in = height_mm / 25.4
		width_dpi = width_px / width_in
		height_dpi = height_px / height_in
		if self.debug_print_flag:
			print('Width: %i px, Height: %i px' % (width_px, height_px))
			print('Width: %i mm, Height: %i mm' % (width_mm, height_mm))
			print('Width: %f in, Height: %f in' % (width_in, height_in))
			print('Width: %f dpi, Height: %f dpi' % (width_dpi, height_dpi))
		# self.canvas = self.master.canvas
		#########################################################################
		#  Set up the plotting frame and controls frame
		#########################################################################
		master.rowconfigure(0, weight=10, minsize=200)
		master.columnconfigure(0, weight=1)
		master.rowconfigure(1, weight=1, minsize=20)
		self.right_frame = tk.Frame(self.master, borderwidth=10, relief='sunken')
		self.right_frame.grid(row=0, column=0, columnspan=1, sticky=tk.N + tk.E + tk.S + tk.W)
		self.matplotlib_width_pixel = self.right_frame.winfo_width()
		self.matplotlib_height_pixel = self.right_frame.winfo_height()
		
		self.right_frame.update()
		
		
	def display_matplotlib_figure_on_tk_canvas(self):
		# Draw a matplotlib figure in a Tk canvas
		self.matplotlib_2d_ax.clear()
		X = np.linspace(0, 2 * np.pi, 100)
		# Y = np.sin(X)
		Y = np.sin(X * np.int((np.random.rand() + .1) * 10))
		self.matplotlib_2d_ax.plot(X, Y)
		self.matplotlib_2d_ax.set_xlim([0, 2 * np.pi])
		self.matplotlib_2d_ax.set_ylim([-1, 1])
		self.matplotlib_2d_ax.grid(True, which='both')
		self.matplotlib_2d_ax.axhline(y=0, color='k')
		self.matplotlib_2d_ax.axvline(x=0, color='k')
		# plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
		# Place the matplotlib figure on canvas and display it
		self.matplotlib_2d_figure_canvas_agg = FigureCanvasAgg(self.matplotlib_2d_fig)
		self.matplotlib_2d_figure_canvas_agg.draw()
		self.matplotlib_2d_figure_x, self.matplotlib_2d_figure_y, self.matplotlib_2d_figure_w, \
		self.matplotlib_2d_figure_h = self.matplotlib_2d_fig.bbox.bounds
		self.matplotlib_2d_figure_w, self.matplotlib_2d_figure_h = int(self.matplotlib_2d_figure_w), int(
			self.matplotlib_2d_figure_h)
		self.photo = tk.PhotoImage(master=self.canvas, width=self.matplotlib_2d_figure_w,
		                           height=self.matplotlib_2d_figure_h)
		# Position: convert from top-left anchor to center anchor
		self.canvas.create_image(self.matplotlib_2d_fig_loc[0] + self.matplotlib_2d_figure_w / 2,
		                         self.matplotlib_2d_fig_loc[1] + self.matplotlib_2d_figure_h / 2, image=self.photo)
		tkagg.blit(self.photo, self.matplotlib_2d_figure_canvas_agg.get_renderer()._renderer, colormode=2)
		self.matplotlib_2d_fig_w, self.matplotlib_2d_fig_h = self.photo.width(), self.photo.height()
		self.canvas.create_text(0, 0, text="Sin Wave", anchor="nw")

	def display_matplotlib_3d_figure_on_tk_canvas(self):
		self.matplotlib_3d_ax.clear()
		r = np.linspace(0, 6, 100)
		temp=np.random.rand()
		theta = np.linspace(-temp * np.pi, temp * np.pi, 40)
		r, theta = np.meshgrid(r, theta)
		X = r * np.sin(theta)
		Y = r * np.cos(theta)
		Z = np.sin(np.sqrt(X ** 2 + Y ** 2))
		surf = self.matplotlib_3d_ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="coolwarm", linewidth=0, antialiased=False);
		# surf = self.matplotlib_3d_ax.plot_surface(X, Y, Z, rcount=1, ccount=1, cmap='bwr', edgecolor='none');
		self.matplotlib_3d_ax.set_xlim(-6, 6)
		self.matplotlib_3d_ax.set_ylim(-6, 6)
		self.matplotlib_3d_ax.set_zlim(-1.01, 1.01)
		self.matplotlib_3d_ax.zaxis.set_major_locator(LinearLocator(10))
		self.matplotlib_3d_ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
		# Place the matplotlib figure on canvas and display it
		self.matplotlib_3d_figure_canvas_agg.draw()
		self.matplotlib_3d_figure_x, self.matplotlib_3d_figure_y, self.matplotlib_3d_figure_w, \
		self.matplotlib_3d_figure_h = self.matplotlib_2d_fig.bbox.bounds
		self.matplotlib_3d_figure_w, self.matplotlib_3d_figure_h = int(self.matplotlib_3d_figure_w), int(
			self.matplotlib_3d_figure_h)
		if self.debug_print_flag:
			print("Matplotlib 3d figure x, y, w, h: ", self.matplotlib_3d_figure_x, self.matplotlib_3d_figure_y,
			      self.matplotlib_3d_figure_w, self.matplotlib_3d_figure_h)
		self.photo = tk.PhotoImage(master=self.canvas, width=self.matplotlib_3d_figure_w,
		                           height=self.matplotlib_3d_figure_h)
		# Position: convert from top-left anchor to center anchor
		self.canvas.create_image(self.matplotlib_3d_fig_loc[0] + self.matplotlib_3d_figure_w / 2,
		                         self.matplotlib_3d_fig_loc[1] + self.matplotlib_3d_figure_h / 2, image=self.photo)
		tkagg.blit(self.photo, self.matplotlib_3d_figure_canvas_agg.get_renderer()._renderer, colormode=2)
		self.matplotlib_3d_fig_w, self.matplotlib_3d_fig_h = self.photo.width(), self.photo.height()

	def key_pressed_callback(self, event):
		self.root.status_bar.set('%s', 'Key pressed')

	def up_arrow_pressed_callback(self, event):
		self.root.status_bar.set('%s', "Up arrow was pressed")

	def down_arrow_pressed_callback(self, event):
		self.root.status_bar.set('%s', "Down arrow was pressed")

	def right_arrow_pressed_callback(self, event):
		self.root.status_bar.set('%s', "Right arrow was pressed")

	def left_arrow_pressed_callback(self, event):
		self.root.status_bar.set('%s', "Left arrow was pressed")

	def shift_up_arrow_pressed_callback(self, event):
		self.root.status_bar.set('%s', "Shift up arrow was pressed")

	def shift_down_arrow_pressed_callback(self, event):
		self.root.status_bar.set('%s', "Shift down arrow was pressed")

	def shift_right_arrow_pressed_callback(self, event):
		self.root.status_bar.set('%s', "Shift right arrow was pressed")

	def shift_left_arrow_pressed_callback(self, event):
		self.root.status_bar.set('%s', "Shift left arrow was pressed")

	def f_key_pressed_callback(self, event):
		self.root.status_bar.set('%s', "f key was pressed")

	def b_key_pressed_callback(self, event):
		self.root.status_bar.set('%s', "b key was pressed")

	def left_mouse_click_callback(self, event):
		self.root.status_bar.set('%s', 'Left mouse button was clicked. ' + 'x=' + str(event.x) + '   y=' + str(
			event.y))
		self.x = event.x
		self.y = event.y
		self.canvas.focus_set()

	def left_mouse_release_callback(self, event):
		self.root.status_bar.set('%s',
		                         'Left mouse button was released. ' + 'x=' + str(event.x) + '   y=' + str(event.y))
		self.x = None
		self.y = None

	def left_mouse_down_motion_callback(self, event):
		self.root.status_bar.set('%s', 'Left mouse down motion. ' + 'x=' + str(event.x) + '   y=' + str(event.y))
		self.x = event.x
		self.y = event.y

	def right_mouse_click_callback(self, event):
		self.root.status_bar.set('%s', 'Right mouse down motion. ' + 'x=' + str(event.x) + '   y=' + str(event.y))
		self.x = event.x
		self.y = event.y

	def right_mouse_release_callback(self, event):
		self.root.status_bar.set('%s',
		                         'Right mouse button was released. ' + 'x=' + str(event.x) + '   y=' + str(event.y))
		self.x = None
		self.y = None

	def right_mouse_down_motion_callback(self, event):
		self.root.status_bar.set('%s', 'Right mouse down motion. ' + 'x=' + str(event.x) + '   y=' + str(event.y))
		self.x = event.x
		self.y = event.y

	def left_mouse_click_callback(self, event):
		self.root.status_bar.set('%s', 'Left mouse button was clicked. ' + 'x=' + str(event.x) + '   y=' + str(
			event.y))
		self.x = event.x
		self.y = event.y

	
	def frame_resized_callback(self, event):
		print("frame resize callback")

	def create_graphic_objects(self):
		self.canvas.delete("all")
		r = np.random.rand()
		self.drawing_objects = []
		for scale in np.linspace(.1, 0.8, 20):
			self.drawing_objects.append(self.canvas.create_oval(int(scale * int(self.canvas.cget("width"))),
			                                                    int(r * int(self.canvas.cget("height"))),
			                                                    int((1 - scale) * int(self.canvas.cget("width"))),
			                                                    int((1 - scale) * int(self.canvas.cget("height")))))

	def redisplay(self, event):
		self.create_graphic_objects()

	def matplotlib_plot_2d_callback(self):
		self.display_matplotlib_figure_on_tk_canvas()
		self.root.status_bar.set('%s', "called matplotlib_plot_2d_callback callback!")

	def matplotlib_plot_3d_callback(self):
		self.display_matplotlib_3d_figure_on_tk_canvas()
		self.root.status_bar.set('%s', "called matplotlib_plot_3d_callback callback!")

	def graphics_draw_callback(self):
		self.create_graphic_objects()
		self.root.status_bar.set('%s', "called the draw callback!")


		
def close_window_callback(root):
	if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
		root.destroy()


main_window = MainWindow(debug_print_flag=False)
# main_window.geometry("500x500")
main_window.wm_state('normal')
main_window.title('Assignment_06 -  Arora')
main_window.minsize(800, 600)
main_window.protocol("WM_DELETE_WINDOW", lambda root_window=main_window: close_window_callback(root_window))
main_window.mainloop()
