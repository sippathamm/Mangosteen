from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler

from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from Roundness import Roundness
from FFT import FFT

import tkinter as tk
import numpy as np
import pandas as pd

import pickle

image_file_path = None
audio_file_path = None
export_result_image = False

KNN = pickle.load(open('Model/KNN.model', 'rb'))
SVM = pickle.load(open('Model/SVM.model', 'rb'))
LR = pickle.load(open('Model/LR.model', 'rb'))
DT = pickle.load(open('Model/DT.model', 'rb'))
XGB = pickle.load(open('Model/XGB.model', 'rb'))
LGBM = pickle.load(open('Model/LGBM.model', 'rb'))
ANN = pickle.load(open('Model/ANN.model', 'rb'))
RF = pickle.load(open('Model/RF.model', 'rb'))
standard_scaler = pickle.load(open('Dataset/normalized_dataset.dataset', 'rb'))

models = {
    'KNN': KNN,
    'SVM': SVM,
    'LR': LR,
    'DT': DT,
    'XGB': XGB,
    'LGBM': LGBM,
    'ANN': ANN,
    'RF': RF,
}

def upload_image():
    global image_file_path
    image_file_path = filedialog.askopenfilename(filetypes=[('Image files', '*.jpg')])
    if image_file_path:
        file_name = image_file_path.split('/')[-1]
        S1_image_name_label.config(text=file_name)

        image = Image.open(image_file_path)
        image.thumbnail((200, 200))
        preview = ImageTk.PhotoImage(image)
        S1_image_preview.config(image=preview)
        S1_image_preview.image = preview


def export_result_image_flag():
    global export_result_image
    export_result_image = S1_checkbox_var.get()


def upload_audio():
    global audio_file_path
    audio_file_path = filedialog.askopenfilename(filetypes=[('Audio files', '*.wav')])
    if audio_file_path:
        file_name = audio_file_path.split('/')[-1]
        S2_audio_name_label.config(text=file_name)


def submit():
    r = Roundness(image_file_path)
    r.analyze()
    roundness = r.get_roundness()
    if export_result_image:
        r.save_result()

    f = FFT(audio_file_path)
    f.analyze()
    max_frequency, max_magnitude = f.get_max_frequency(), f.get_max_magnitude()

    weight = float(S3_weight_entry.get())
    lower_petal = int(S4_lower_petal_entry.get())

    model = models[S5_model_var.get()]

    X = pd.DataFrame({'Weight (g)': [weight],
                      'Roundness': [roundness],
                      'Lower Petal': [lower_petal],
                      'Max Frequency (Hz)': [max_frequency],
                      'Max Magnitude': [max_magnitude]})
    X = standard_scaler.transform(X)

    prediction = predict(model, X)

    label = 'Low Sweetness' if prediction[0] == 0 else 'High Sweetness'

    prediction_label.config(text=f'Prediction: {label} (Class {prediction[0]})')


def predict(model, X):
    y_predict = model.predict(X)
    return y_predict


root = tk.Tk()
root.title('Classifying the Sweetness of Mangosteen')
root.geometry('800x700')

S1_title = ttk.Label(root, text='Select Image 📷:')
S1_title.pack()
S1_image_preview = ttk.Label(root, text='[Not selected]')
S1_image_preview.pack()
S1_image_name_label = ttk.Label(root, text='')
S1_image_name_label.pack()
S1_image_browse = ttk.Button(root, text='Browse', command=upload_image)
S1_image_browse.pack()

S1_checkbox_var = tk.BooleanVar()
S1_checkbox = ttk.Checkbutton(root, text='Export Roundness Detection',
                              variable=S1_checkbox_var, command=export_result_image_flag)
S1_checkbox.pack()

separator1 = ttk.Separator(root, orient='horizontal')
separator1.pack(fill='x', padx=150, pady=10)

S2_title = ttk.Label(root, text='Select Audio 🎧️:')
S2_title.pack()
S2_audio_name_label = ttk.Label(root, text='[Not selected]')
S2_audio_name_label.pack()
S2_audio_browse = ttk.Button(root, text='Browse', command=upload_audio)
S2_audio_browse.pack()

separator2 = ttk.Separator(root, orient='horizontal')
separator2.pack(fill='x', padx=150, pady=10)

S3_weight_label = ttk.Label(root, text='Enter Weight (g) ⚖️:')
S3_weight_label.pack()
S3_weight_entry = ttk.Entry(root)
S3_weight_entry.pack()

separator3 = ttk.Separator(root, orient='horizontal')
separator3.pack(fill='x', padx=150, pady=10)

S4_lower_petal_label = ttk.Label(root, text='Enter Lower Petal ✏️️:')
S4_lower_petal_label.pack()
S4_lower_petal_entry = ttk.Entry(root)
S4_lower_petal_entry.pack()

separator4 = ttk.Separator(root, orient='horizontal')
separator4.pack(fill='x', padx=150, pady=10)

S5_model_label = ttk.Label(root, text='Select ML Model 💻:')
S5_model_label.pack()
S5_model_var = tk.StringVar(root)
S5_model_menu = ttk.OptionMenu(root, S5_model_var, list(models.keys())[0], *list(models.keys()))
S5_model_menu.pack()

predict_button = ttk.Button(root, text='Predict', command=submit)
predict_button.pack()

prediction_label = ttk.Label(root, text='', font=('Helvetica', 16))
prediction_label.pack(pady=(10, 0))

root.mainloop()
