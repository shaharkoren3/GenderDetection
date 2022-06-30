Instructions:

1. Check if you have Python installed:
   - open cmd
   - type python
2. If Python not installed, install from: https://www.python.org/downloads/
3. Create a new folder called gender_detection in c: (if you don't have a drive on C, use another drive)
   - cd\
   - mkdir gender_detection
   - cd gender_detection
4. From my drive, download the dataset (link: https://drive.google.com/drive/folders/1eylljVemM8Wg7l-7RbyPwGq2M6XrXcB6 ) and copy to c:\gender_detection
5. Install virtualenv (if virtualenv.exe is not in the path, use full path from where virtualenv.exe was installed):
   - pip install virtualenv --user
6. Create a virtualenv in gender_detection
   virtaulenv venv
7. Activate the virtual env
   - venv\Scripts\activate.bat
8. To deactivate use:
   - deactivate
9. Install win wrapper:
   - pip install virtualenvwrapper-win
10. Install dependencies:
   pip install numpy
   pip install Pillow
   pip install tensorflow
   pip install matplotlib
   pip install opencv-python
   pip install sklearn
11. Run the app using:
   - python source\main.py
   

Please note:

- This code can be run from cmd. However, it creates graphs which will only be visible if you use a 
  graphical environment. We recommend using an IDE, for example Pycharm, to run the code from, in 
  order to see all the graph which are created
- The images in the predict_options folder can be used to test the model  