import pandas as pd
import cleaning_data
import eda
import ML_model

# reading the training data file
comments_df = pd.read_csv('dataset/train.csv')
print(comments_df.columns.values)

# Exploratory Data Analysis
eda.exploratory_data_analysis(comments_df)

# Cleaning the data
comments_df = cleaning_data.data_cleaning(comments_df)

# Training the model
ML_model.classification_model_train(comments_df)

print("The model is now trained...")
print("====================================================================================================")
run_test_file = int(input("Press 1 to see results on test data: "))

if run_test_file == 1:
    # reading the testing data file
    comments_df = pd.read_csv('dataset/test.csv')
    print(comments_df.columns.values)

    # Cleaning the data
    comments_df = cleaning_data.data_cleaning(comments_df)

    # Testing the model
    ML_model.classification_model_test(comments_df)