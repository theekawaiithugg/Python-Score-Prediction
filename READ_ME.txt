This predicitin software come with a demo game of snake to show how to use the gui.

The game also creates the files necessary to Run the prediction program.
The game data file is saved as a csv file and placed in the file 
you save the program game in.

Please adjust accordingly the Prediction file is created to skip the first line in
 the csv file, to be sure to exclude labels from the file. 

When creating your data files be sure to pull the portion in the beggining of the
pythongame 
*Lines 14 to 37 cleaned up file
*edit line 18 to label the data accordingly
*edit line 20 or the expected_fields variable to match

Be sure to add a reset function like line 139
Be sure to add from the game over definition the with function sta rting on line 167
to log data properly after a game has ended and
*edit line 173 to match line 18 labels
*edot 175 to match variable to line 173

THE PREDICTION FILE
It is to use 3 different versions of the linear regression model. The lower the
Python version the lower the accuracy and the higher the version the higher the 
accuracy.

First model is XGBoost based.
Second Model is sklearn based
Third model is TensorFlow baded.

The mse and rmse valuses will give you how accurate the data is based off of the 
set of data given. They will print our prediction scores of each model in terminal
 of use.