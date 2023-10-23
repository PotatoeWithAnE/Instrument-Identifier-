import tkinter
import tkinter as tk
from tkinter import filedialog as fd
from PIL import Image,ImageTk
import cv2
import numpy as np
import tensorflow as tf
import os


#creating a dataset
def make_dataset(directory):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode='rgb',
        batch_size=32,
        image_size=(256, 256),
        shuffle=True
    )
    return dataset


#set data_dir to 'data'
train_data_dir = 'data_train'
test_data_dir = 'data_test'
train_dataset = make_dataset(train_data_dir)
test_dataset = make_dataset(test_data_dir)


if os.path.isfile('instrument_sorter_model.keras'):
    model = tf.keras.models.load_model('instrument_sorter_model.keras')
    print("model loaded!")

else:
    model = tf.keras.models.Sequential([
        #setting sizes and augumenting data
        tf.keras.layers.Input(shape=(256, 256, 3)),
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),

    #convolutional layers of increasing depth
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

    #flatten 2D->1D array
        tf.keras.layers.Flatten(),

    #number of neurons in final layers
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )





#training model
test_loss, test_accuracy = model.evaluate(test_dataset)


if test_accuracy < .74:
    model.fit(train_dataset, epochs=12)
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)
    



#saving model for running again
model.save('instrument_sorter_model.keras')
print("model saved!")


#setting up tkinter input window
window_input = tk.Tk()
window_input.geometry("500x400+100+100")
window_input.resizable(False,False)



#creating input window to extract image file location
target_image_path = ""
def open_file_dialog(): #command for using the file dialog to extract img directory
    global target_image_path
    image_path = fd.askopenfilename()
    print(image_path)
    selected_directory_label.config(text=f"{image_path}")
    target_image_path = image_path
    confirm_button.config(command = window_input.destroy) #confirm button now destroys current window


#clippy
clippy_image = Image.open("clippy.png")
clippy_image = clippy_image.resize((500,400))
clippy_image = ImageTk.PhotoImage(clippy_image)

clippy_label = tk.Label(window_input, image=clippy_image)
clippy_label.place(x=0, y=0, relwidth=1, relheight=1)

window_input.title("Choose a File")

#setting up instructionss
instructions_label1 = tk.Label(text = "Press the first button and choose an",background="#ffffcb")
instructions_label2 = tk.Label(text = "image file. Then, I'll tell you if",background="#ffffcb")
instructions_label3 = tk.Label(text = "it's a guitar, piano or a drum set!",background="#ffffcb")

instructions_label1.place(x=25,y=35)
instructions_label2.place(x=25,y=60)
instructions_label3.place(x=25,y=85)

#setting up buttons
open_directory_button = tk.Button(window_input, text="Select File", command=open_file_dialog)
selected_directory_label = tk.Label(window_input, text="Selected File: None",background="#ffffcb")
confirm_button = tk.Button(window_input, text="Confirm and Close")

open_directory_button.place(y=150, x=25)
confirm_button.place(y=150,x=110)
selected_directory_label.place(y=120,x =45)



window_input.mainloop()


#prediction of input image
img_mat = cv2.imread(target_image_path)
img_mat = cv2.resize(img_mat, (256, 256))
img = img_mat
img_mat = img_mat.reshape((1, 256, 256, 3))
img_mat = img_mat.astype('float32')
prediction = model.predict(img_mat)
predicted_classes = np.argmax(prediction, axis=-1)


categories = ('drums',"guitar",'piano')
print(prediction)
print(f"This image is: {categories[predicted_classes[0]]}")

##setup output window
window_output = tk.Tk()
window_output.geometry("500x400+100+100")

window_output.title("Results")

#Wingardeium Leviousa
output_info = tk.Label(window_output,text = f"The Instrument is: {categories[predicted_classes[0]]}")
output_info.pack()
window_input.mainloop()