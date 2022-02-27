# Caroline Cocca
# February 27, 2022
# Creating AI-Enabled Systems
# Assignment 4
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


# Read in the cars csv at the given path.
# Return as pandas dataframe.
def get_cars(path):
    cars_df = pd.read_csv(path)
    return cars_df


# The transmission column is the only column with 2 unique values.
# Map these values to a boolean column using pandas.map
def get_boolean_transmission(df):
    transmission_map = {'automatic': 0, 'mechanical': 1}
    df['transmission'] = df['transmission'].map(transmission_map)
    return df


# For columns with less than 10 unique values, convert to one-hot encoding.
# This would be inappropriate for columns with many unique values,
# due to the resulting generated columns causing the amount of features,
# and thus input nodes, to explode in length if there are many unique values
# being encoded.
# pd.get_dummies returns a dataframe containing the boolean cols generated from the
# one-hot encoding mapping, where a column exists for each unique value.
# drop_first will drop one of these columns in order to save memory.
# This column will be inferred as true when all other remaining cols equal 0/false.
def get_dummies(df):
    engine_type_d = pd.get_dummies(df.engine_type, prefix="engine_type", drop_first=True)
    state_d = pd.get_dummies(df.state, prefix="state", drop_first=True)
    drivetrain_d = pd.get_dummies(df.drivetrain, prefix="drivetrain", drop_first=True)
    engine_fuel_d = pd.get_dummies(df.engine_fuel, prefix="engine_fuel", drop_first=True)
    location_region_d = pd.get_dummies(df.location_region, prefix="region", drop_first=True)
    df = pd.concat([df, engine_type_d, state_d, drivetrain_d, engine_fuel_d,
                    location_region_d], axis=1)
    df = df.drop(columns=['engine_type', 'state', 'drivetrain', 'engine_fuel',
                          'location_region'], axis=1)
    return df


# For columns with more than 10 unique values, convert to ordinal encoding.
# Ordinal encoding results in a unique numerical value being associated with each
# unique categorical value from the original categorical column.
# The drawback to this method is that there is an implied order to the
# categorical values where no true order may exist.
# However, this method saves memory/reduces the feature list length.
def get_ordinal_encodings(df):
    df['manufacturer_name'] = OrdinalEncoder().fit_transform(df[['manufacturer_name']])
    df['model_name'] = OrdinalEncoder().fit_transform(df[['model_name']])
    df['color'] = OrdinalEncoder().fit_transform(df[['color']])
    df['body_type'] = OrdinalEncoder().fit_transform(df[['body_type']])
    return df


if __name__ == '__main__':
    # read the cars dataset into a pandas dataframe
    cars = get_cars("cars.csv")

    # see a summary of the original columns
    print("-----------------------------------------------")
    print("Original columns")
    print()
    cars.info()
    cat_cols = ['manufacturer_name', 'model_name', 'transmission', 'color',
                'engine_fuel', 'engine_type', 'body_type', 'state', 'drivetrain',
                'location_region']
    print("-----------------------------------------------")
    print()

    # check how many unique values each categorical column contains
    print("-----------------------------------------------")
    print("Categorical column unique value counts")
    print()
    for key in cat_cols:
        print(key + ": " + str(len(cars[key].unique())))
    print("-----------------------------------------------")
    print()

    # transform the categorical columns
    cars = get_boolean_transmission(cars)
    cars = get_dummies(cars)
    cars = get_ordinal_encodings(cars)

    # output the final columns summary
    print("-----------------------------------------------")
    print("Final columns")
    print()
    cars.info()
    print()
    print("No more object columns!")
    print("-----------------------------------------------")
