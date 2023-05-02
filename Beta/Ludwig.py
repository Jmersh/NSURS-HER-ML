import pandas as pd
from sklearn.model_selection import train_test_split
from ludwig.api import LudwigModel

# Load data from the CSV file
data = """TrainData.csv"""  # Replace this with your actual CSV data
csv_data = pd.read_csv(pd.StringIO(data), sep=',')

# Split the data into training and testing sets
train_data, test_data = train_test_split(csv_data, test_size=0.2, random_state=42)
train_data.to_csv('train_dataL.csv', index=False)
test_data.to_csv('test_dataL.csv', index=False)

# Define the model configuration
model_definition = {
    "input_features": [
        {"name": col, "type": "numerical"} for col in train_data.columns if col != 'Energy'
    ],
    "output_features": [
        {"name": "Energy", "type": "numerical"}
    ]
}

# Train the model
ludwig_model = LudwigModel(model_definition)
train_stats = ludwig_model.train(pd.read_csv('train_dataL.csv'))

# Make predictions on the test set
predictions = ludwig_model.predict(pd.read_csv('test_dataL.csv'))

# Calculate the mean squared error of the predictions
mse = ((predictions['Energy_predictions'] - test_data['Energy'].reset_index(drop=True)) ** 2).mean()

print(f"Mean Squared Error: {mse}")