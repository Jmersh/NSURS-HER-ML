using CSV, DataFrames, Random, Dates, DecisionTree, Statistics

# Read CSV file
df = CSV.read("Data.csv", DataFrame)


# Select the last column as the target
target_column = names(df)[lastindex(df, 2)]

# Split the data into training and testing sets
Random.seed!(123)
train_ratio = 0.8
nrows = nrow(df)
train_indices = 1:div(nrows, 5)
test_indices = div(nrows, 5) + 1:nrows

train_data = df[train_indices, :]
test_data = df[test_indices, :]

# Extract the features and target from the data
train_features = Matrix(train_data[:, 1:lastindex(df, 2) - 1])
train_target = train_data[:, target_column]

test_features = Matrix(test_data[:, 1:lastindex(df, 2) - 1])
test_target = test_data[:, target_column]

# Perform random forest regression
model = build_forest(train_target, train_features, 10, 10)

# Measure training time
training_time = @elapsed begin
    train_predictions = apply_forest(model, train_features)
end
training_time_ms = training_time * 1000  # Convert to milliseconds

# Report the results
println("Training Time: ", training_time_ms, " milliseconds")
# Measure prediction time
@elapsed begin
    test_predictions = apply_forest(model, test_features)
end

# Measure prediction time
prediction_time = @elapsed apply_forest(model, test_features)
prediction_time_ms = prediction_time * 1000  # Convert to milliseconds

# Report the results
println("Prediction Time: ", prediction_time_ms, " milliseconds")



# Calculate RMSE and MSE
train_rmse = sqrt(mean((train_predictions .- train_target).^2))
test_rmse = sqrt(mean((test_predictions .- test_target).^2))

train_mse = mean((train_predictions .- train_target).^2)
test_mse = mean((test_predictions .- test_target).^2)

# Report the results
# println("Training Time: ", Millisecond(train_end_time - train_start_time))
# println("Prediction Time: ", Millisecond(predict_end_time - predict_start_time))
println("RMSE (Training): ", train_rmse)
println("RMSE (Testing): ", test_rmse)
println("MSE (Training): ", train_mse)
println("MSE (Testing): ", test_mse)