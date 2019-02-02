Code Location: https://github.com/thatguyandy27/CS-7641/tree/master/project1

Run this code requirements

1. Python3
2. anaconda libraries
    - scipy
    - pandas
    - numpy
    - matplotlib
3. Put the data in the ./data/ folder
    - white wine dataset
        - Download from https://archive.ics.uci.edu/ml/datasets/Wine+Quality
        - Put the winequality-white.csv file in the ./data/wine/ folder.
    - diabetes dataset
        - Download from https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set
        - Remove the first 24 lines from the file (until @data)
        - Put the data.text file in the ./data/diabetes_renop/ folder.
4. Run 'python main.py'
    - To not run some of the algorithms, in main.py set the corresponding flag to False
    - Example nn_run = False
    - Neural Networks take a long time to run
5. Results are sent to the ./output/ folder
6. Images are sent to the ./output/images/
