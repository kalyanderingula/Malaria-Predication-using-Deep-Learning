					MALARIA PREDICTION USING DEEP LEARNING
Predicting malaria using deep learning is a neural network that analyzes microscopic blood smear images to detect parasites, aiding in early diagnosis and treatment.

Pre-requisites:
	Install the below-required software stack and python libraries before running the app.
		1. Python
		2. CSS
		3. HTML
		4. Dataset
	Python Libraries:
		1. Tensorflow
		2. Numpy
		3. Flask
		4. Keras
		5. Matplotlib

Process followed to build the App:
Model.py:-

* To build the model used TensorFlow and the below modules  to run the CNN model
     from the Tensorflow import keras
     from Keras.model import Sequential
     
1. The Convolution2D layer will help to divide the image into small pieces based on the use matrix then input shape for that image length and height for it then activation is relu which is known as rectified linear unit.

2. MaxPooling2D will merge the image pixel using the max of matrix values
3. Flattern this layer will help to convert the 2d or 3d into single d
4. Dense layer will merge all pixels into a single Image this is how to train the model then inside the dense use the activation='softmax' this model will move and run smoothly

* ImageDataGenerator will help to resize and reshape the image so that we can suitable for train and test

     history=classifier.fit_generator(training_set,steps_per_epoch ,epochs )
     
* fit_generator will help to run and train the model

* steps_per_epoch will help to run each epoch and will train the image that many times set by the developer.

* We stored the train data in the H5 file, the Hierarchical Data Format version 5 (HDF5), which is an open-source file format that supports large, complex, heterogeneous data.

* ggplot is used to display the accuracy, actual accuracy, loss, and actual loss graph and save it as a PNG format

app.py
------
* To build the app.py, used the modules like keras, flask, numpy and load_model
* Flask application, we are using a pre-trained Convolutional Neural Network (CNN) model to classify images of malaria parasites as either parasitized or uninfected. The model is trained using the Keras library in Python.

* The application has two main routes:

1. The home page ('/'): This is the default route when the application is accessed. It displays an HTML form that allows users to upload an image file.

2. The prediction route ('/predict'): This route is triggered when the user submits the form on the home page. The uploaded image file is processed and passed to the prediction function. The prediction function preprocesses the image, feeds it into the trained model, and returns the predicted class (either parasitized or uninfected). The predicted class is then displayed on a separate HTML page.

* The prediction function uses the following steps:

1. Load the pre-trained model using the load_model function from the Keras library.

2. Define the classes that the model can predict. In this case, the classes are 'Parasitized' and 'Uninfected'.

3. Define a function to preprocess the image before feeding it into the model. This function resizes the image to the size expected by the model (64x64 pixels) and converts it into an array that can be fed into the model.

4. Define a function to make predictions using the pre-trained model. This function takes the path of an image file as input, preprocesses the image using the preprocessing function, feeds the preprocessed image into the model, and returns the predicted class.

5. In the prediction route, check if the request method is POST (i.e., the user has submitted the form). If the request method is POST, save the uploaded image file to the server, preprocess it using the preprocessing function, and pass it to the prediction function. The predicted class is then displayed on a separate HTML page.

6. The application uses the Flask web framework to handle user requests and render HTML templates. The HTML templates are stored in a folder named 'templates' in the root directory of the application. The templates include a form for uploading image files and a page to display the predicted class.

Steps to Run /Execute the APP:
1. Clone the git repo using the below command:
     git clone https://github.com/kalyanderingula/Malaria-Predication-Using-Deep-Learning-.git

2. Go to the cloned repo folder:
   cd Malaria-Predication-Using-Deep-Learning-

3. Download the data sets from https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria to Malaria-Predication-Using-Deep-Learning- directory.
4. Run the application with the below command in your terminal:
     python3 app.py
5. This will start the Flask development server. You can then access the application by navigating to the URL displayed in your terminal (e.g., http://127.0.0.1:5000/).
6. On App UI, follow the steps mentioned in report.pdf to upload images and predict data using the model.

