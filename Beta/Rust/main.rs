// Various distance metrics
use smartcore::metrics::distance::*;
use smartcore::model_selection::train_test_split;
use std::error::Error;
use csv::Reader;
use std::fs::File;
use ndarray::{Array2};
use std::time::Instant;
use smartcore::neighbors::knn_regressor::KNNRegressor;
use smartcore::metrics::mean_squared_error;
use smartcore::neighbors::knn_regressor::KNNRegressorParameters;
use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;
use std::io::Write;

// Turn Rust vector-slices with samples into a matrix
fn main() -> Result<(), Box<dyn Error>> {
    // Read the dataset from a CSV file
    let file = File::open(r"TrainData.csv)?;
    let mut reader = Reader::from_reader(file);

    // Initialize a vector to store the rows of the dataset
    let mut rows: Vec<Vec<f64>> = Vec::new();

    // Iterate over each record in the CSV file
    for result in reader.records() {
        let record = result?;
        let row: Vec<f64> = record
            .iter()
            .skip(1)
            .filter_map(|field| field.parse().ok())
            .collect();
        rows.push(row);
    }

    // Determine the maximum number of columns in the rows
    let num_columns = rows.iter().map(|row| row.len()).max().unwrap_or(0);

    // Convert the vector of rows into an ndarray::Array2
    let x = Array2::from_shape_vec((rows.len(), num_columns), rows.clone().into_iter().flatten().collect())
        .unwrap_or_else(|err| panic!("Failed to create features array: {}", err));
    let y = x.column(num_columns - 1).to_owned();


    // Split the dataset into training and testing sets
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.25, true, None);

    // Print the shapes of the training and testing sets
    println!(
        "X train: {:?}, y train: {}, X test: {:?}, y test: {}",
        x_train.shape(),
        y_train.len(),
        x_test.shape(),
        y_test.len()
    );

    // K Nearest Neighbor Regressor
    let start_training_knn = Instant::now();
    let y_hat_knn = KNNRegressor::fit(
        &x_train,
        &y_train,
        KNNRegressorParameters::default().with_distance(Distances::euclidian()),
    )
        .and_then(|knn| knn.predict(&x_test))
        .unwrap();

    // Random Forest Regressor
    let start_training_rfr = Instant::now();
    let RFRRegressor = RandomForestRegressor::fit(&x_train, &y_train, Default::default()).unwrap();
    let end_training_rfr = Instant::now();
    let start_prediction_rfr = Instant::now();
    let y_hat_rfr = RFRRegressor.predict(&x_test).unwrap(); // use the same data for prediction
    let end_prediction_rfr = Instant::now();

    
    // Create a file to save the output
    let mut output_file = File::create("output.txt")?;

    // Print the shapes of the training and testing sets
    let output = format!(
        "X train: {:?}, y train: {}, X test: {:?}, y test: {}\n",
        x_train.shape(),
        y_train.len(),
        x_test.shape(),
        y_test.len()
    );
    print_and_save(&output, &mut output_file)?;

    // Print the ground truth and predicted values
    let output = format!("y_test: {:?}\n", y_test);
    print_and_save(&output, &mut output_file)?;
    let output = format!("Predictions KNN: {:?}\n", y_hat_knn);
    print_and_save(&output, &mut output_file)?;
    let output = format!("Predictions RFR: {:?}\n", y_hat_rfr);
    print_and_save(&output, &mut output_file)?;

    // Calculate test errors
    let output = format!("MSE KNN: {}\n", mean_squared_error(&y_test, &y_hat_knn));
    print_and_save(&output, &mut output_file)?;
    let output = format!("RMSE KNN: {}\n", mean_squared_error(&y_test, &y_hat_knn).sqrt());
    print_and_save(&output, &mut output_file)?;
    let output = format!("MSE RFR: {}\n", mean_squared_error(&y_test, &y_hat_rfr));
    print_and_save(&output, &mut output_file)?;
    let output = format!("RMSE RFR: {}\n", mean_squared_error(&y_test, &y_hat_rfr).sqrt());
    print_and_save(&output, &mut output_file)?;

    // Print the elapsed times
    let training_time_rfr = end_training_rfr.duration_since(start_training_rfr);
    let prediction_time_rfr = end_prediction_rfr.duration_since(start_prediction_rfr);
    let output = format!("Training Time RFR: {:?}\n", training_time_rfr);
    print_and_save(&output, &mut output_file)?;
    let output = format!("Prediction Time RFR: {:?}\n", prediction_time_rfr);
    print_and_save(&output, &mut output_file)?;

    Ok(())
}

fn print_and_save(output: &str, file: &mut File) -> Result<(), Box<dyn Error>> {
    // Print the output to stdout
    print!("{}", output);

    // Save the output to the file
    file.write_all(output.as_bytes())?;

    Ok(())
}