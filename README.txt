**Personalised Advertisemnt**
1. Open command Prompt and go to directory of where the project folder is located 
2. Setup a virtual environment by running 
python -m venv env
 and activate with (.\env\Scripts\activate) for windows and (source env\bin\activate) For MacOs
3. run command pip install -r requirements.txt to install all dependencies.
4. Run command python app.py and give camera permission if asked.



# Project Components:
- haarcascade is for face detection.
- camera.py is the module for video streaming, frame capturing, prediction and recommendation which are passed to main.py.
- main.py is the main flask application file.
- index.html in 'templates' directory is the web page for the application. Basic HTML and CSS.
- utils.py is an utility module for video streaming of web camera with threads to enable real time detection.
- train.py is the script for image processing and training the model.