import pandas as pd
import os
import csv
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, font

# Set the current working directory to the script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# Define the column names upfront
column_names = ['Timestamp', 'PlayerName', 'Score', 'SnakeLength', 'DistanceToFood', 'ScoreIncrease', "GameDuration"]

# Open the CSV file
with open("snake_scores_clean.csv", 'r') as f:
    # Create a CSV reader object
    reader = csv.reader(f)

     # Create a set to store unique names
    unique_names = set()

    # Skip the header row
    next(reader)

    # Loop through the rows in the CSV file
    for row in reader:
        # Add the name to the set (automatically handles duplicates)
        unique_names.add(row[1])

# Convert the set back to a list if needed
unique_names_list = list(unique_names)

# Print the unique names
print(unique_names_list) 

class UserScorePredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("User Score Prediction")
        self.root.configure(bg='light blue')  # Setting background color of the window
        self.custom_font = font.Font(family='Helvetica', size=12)
        
        # Load the dataset
        self.data = pd.read_csv("snake_scores_clean.csv", names=column_names, header=None, skiprows=1)
        
        # Label for Bropdow Box entry
        self.label = tk.Label(root, text="Pick your Player to analyze", bg='light blue', fg='black', font=self.custom_font)
        self.label.grid(row=0, column=1, pady=10, padx=10)
        
        #drop down list of users
        self.selected_name = tk.StringVar() 
        self.combobox  = ttk.Combobox(root, width = 27, textvariable = self.selected_name) 
        self.combobox['values'] = unique_names_list
        self.combobox.grid(row=1, column=1, pady=10, padx=10)  # This will display the combobox in the GUI
        # Label for Bropdow Box entry
        self.label = tk.Label(root, text="Three diffirent Linear Regresion Models", bg='light blue', fg='black', font=self.custom_font)
        self.label.grid(row=2, column=1, pady=10, padx=10)
        #3 Different Linear Regression style Button
        self.Analysis1 = tk.Button(root,text='XGBoost Model',command=self.analyze_1_funct, bg='navy', fg='white', font=self.custom_font)
        self.Analysis2 = tk.Button(root, text='Sklearn Model', command=self.Analsis_2_SK,bg='navy', fg='white', font=self.custom_font)
        self.Analysis3 = tk.Button(root, text='Tensor Flow Model',command=self.Analsis_3_TF, bg='navy', fg='white', font=self.custom_font)

        self.Analysis1.grid(row=3, column=0, pady=10, padx=10)
        self.Analysis2.grid(row=3, column=1, pady=10, padx=10)
        self.Analysis3.grid(row=3, column=2, pady=10, padx=10)
    
    def analyze_1_funct(self):
        selected_player = self.selected_name.get()
        if not selected_player:
            messagebox.showerror("Error", "No player selected")
            return
        # Load the dataset
        data = pd.read_csv("snake_scores_clean.csv", names=column_names, header=None, skiprows=1)
        player_data = data[data['PlayerName'] == selected_player]

        if player_data.empty:
            messagebox.showinfo("Result", "No data for selected player.")
            return
        
        # Assuming the 'SnakeLength', 'DistanceToFood', and 'ScoreIncrease' are the features
        # and 'Score' is the target variable as before
        X = player_data[['SnakeLength', 'DistanceToFood', 'ScoreIncrease']].values
        y = player_data['Score'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Splitting the dataset into training and testing set
        model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)  # Initialize the XGBoost regressor model
        model.fit(X_train, y_train) # Fit the model
        predictions = model.predict(X_test)  # Predictions on test set
        mse = mean_squared_error(y_test, predictions) # Evaluate the model
        rmse = np.sqrt(mse)
        print("Model Performance:Root Mean Squared Error: ", rmse)

        # Predicting the score for a new set of values
        new_values = np.array([[10, 5, 2]])  # The new values for 'SnakeLength', 'DistanceToFood', 'ScoreIncrease'
        predicted_score = model.predict(new_values)
        print("Predicted Score: ", predicted_score[0])
        return predicted_score
    
    def Analsis_2_SK(self):
        selected_player = self.selected_name.get()
        if not selected_player:
            messagebox.showerror("Error", "No player selected")
            return
         # Assuming you've already loaded the dataset and it's stored in self.data
        player_data = self.data[self.data['PlayerName'] == selected_player]

        if player_data.empty:
            messagebox.showinfo("Result", "No data for selected player.")
            return
        snake_length = player_data['SnakeLength'].mean()
        game_duration = player_data['GameDuration'].mean()
        score_increase = player_data['ScoreIncrease'].mean()
        distance_to_food= player_data['DistanceToFood'].mean()
        # Prepare the dataset
        X = player_data[['SnakeLength', 'DistanceToFood', 'ScoreIncrease', 'GameDuration']].values
        y = player_data['Score'].values

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model training
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Model evaluation
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        # Present the results
        print("Result", f"Root Mean Squared Error: {rmse:.2f}")

        # Predict future score based on input
        prediction_input = np.array([[snake_length, game_duration,score_increase,distance_to_food]])
        future_score = model.predict(prediction_input)
        print("Predicted Score", f"The predicted score is: {future_score[0]:.2f}")
    
    def Analsis_3_TF(self):
        selected_player = self.selected_name.get()
        if not selected_player:
            messagebox.showerror("Error", "No player selected")
            return
         # Assuming you've already loaded the dataset and it's stored in self.data
        player_data = self.data[self.data['PlayerName'] == selected_player]

        if player_data.empty:
            messagebox.showinfo("Result", "No data for selected player.")
            return
        # For the features, use 'SnakeLength', 'DistanceToFood', and 'ScoreIncrease'
        X = player_data[['SnakeLength', 'DistanceToFood', 'ScoreIncrease']].values  # Use .values to get NumPy array
        y = player_data['Score'].values  # Use .values to get NumPy array

        # Splitting the dataset into training and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalize features
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dense(64, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        # Fit the model
        model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100)

        # Evaluate the model using the test set
        mse = model.evaluate(X_test_scaled, y_test)
        print(f"Test MSE: {mse}")
        # Predict future scores using the test set or new data
        predictions = model.predict(X_test_scaled)
        # Optionally, show the first few predicted scores
        print("Sample Predicted Scores:", predictions[:5])

# Create the main window and pass it to the SentimentAnalyzerApp class
root = tk.Tk()
app = UserScorePredictionApp(root)
root.mainloop()
