import json
import joblib
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score


def load_dataset(path):
    return pd.read_csv(path)


def get_shape(df):
    # Teacher requested: shape property
    return df.shape


def get_dtypes_html(df):
    # Teacher requested: dtypes property
    details = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isna().sum().values,
        "Unique Values": df.nunique().values
    })
    return details.to_html(classes="data-table", index=False, border=0)

def get_head_html(df, n):
    # Teacher requested: head function
    return df.head(n).to_html(classes="data-table", index=False, border=0)


def get_tail_html(df, n):
    # Teacher requested: tail function
    return df.tail(n).to_html(classes="data-table", index=False, border=0)


def get_describe_html(df):
    # Teacher requested: describe function
    return df.describe(include="all").fillna("").to_html(classes="data-table", border=0)


def drop_empty_rows(path):
    # Teacher requested: dropna function
    df = pd.read_csv(path)
    old_rows = df.shape[0]
    df = df.dropna()
    df.to_csv(path, index=False)
    return old_rows, df.shape[0], df.shape[1]


def convert_numeric_columns(df):
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="ignore")
    return df


def transform_with_get_dummies(df, feature_columns):
    # Teacher requested: get_dummies function
    X = df[feature_columns].copy()
    X = pd.get_dummies(X, drop_first=True)
    return X


def transform_with_ordinal_encoder(df, feature_columns):
    # Teacher requested: OrdinalEncoder fit_transform function
    X = df[feature_columns].copy()

    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    X_numeric = X[numeric_columns].copy()

    encoder = None
    if categorical_columns:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoded_values = encoder.fit_transform(X[categorical_columns])
        X_encoded = pd.DataFrame(
            encoded_values,
            columns=categorical_columns,
            index=X.index
        )
        X_final = pd.concat([X_numeric, X_encoded], axis=1)
    else:
        X_final = X_numeric

    return X_final, encoder, categorical_columns, numeric_columns


def apply_ordinal_encoder_for_prediction(input_df, encoder, categorical_columns, numeric_columns):
    numeric_part = input_df[numeric_columns].copy() if numeric_columns else pd.DataFrame(index=input_df.index)

    if categorical_columns:
        encoded_values = encoder.transform(input_df[categorical_columns])
        encoded_part = pd.DataFrame(
            encoded_values,
            columns=categorical_columns,
            index=input_df.index
        )
        return pd.concat([numeric_part, encoded_part], axis=1)

    return numeric_part


def get_algorithm(algorithm_type, algorithm_name):
    if algorithm_type == "regression":
        if algorithm_name == "random_forest_regressor":
            return RandomForestRegressor(n_estimators=100, random_state=42)
        return LinearRegression()

    if algorithm_type == "classification":
        if algorithm_name == "random_forest_classifier":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        return LogisticRegression(max_iter=1000)

    if algorithm_type == "clustering":
        return KMeans(n_clusters=3, random_state=42, n_init="auto")

    raise ValueError("Invalid algorithm type")


def train_model(
    df,
    feature_columns,
    target_column,
    algorithm_type,
    algorithm_name,
    preprocessing_method,
    model_path
):
    df = df.copy()
    df = df.dropna()
    df = convert_numeric_columns(df)

    model_object = {
        "preprocessing_method": preprocessing_method,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "algorithm_type": algorithm_type,
        "algorithm_name": algorithm_name,
        "model": None,
        "encoder": None,
        "categorical_columns": None,
        "numeric_columns": None,
        "dummy_columns": None,
        "scaler": None
    }

    if preprocessing_method == "get_dummies":
        X = transform_with_get_dummies(df, feature_columns)
        model_object["dummy_columns"] = X.columns.tolist()
    else:
        X, encoder, categorical_columns, numeric_columns = transform_with_ordinal_encoder(df, feature_columns)
        model_object["encoder"] = encoder
        model_object["categorical_columns"] = categorical_columns
        model_object["numeric_columns"] = numeric_columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model_object["scaler"] = scaler

    algorithm = get_algorithm(algorithm_type, algorithm_name)

    score = None

    if algorithm_type in ["regression", "classification"]:
        if not target_column:
            raise ValueError("Target column is required for regression and classification.")

        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=0.2,
            random_state=42
        )

        algorithm.fit(X_train, y_train)
        predictions = algorithm.predict(X_test)

        if algorithm_type == "classification":
            score = accuracy_score(y_test, predictions)
        else:
            score = r2_score(y_test, predictions)

    else:
        algorithm.fit(X_scaled)

    model_object["model"] = algorithm
    joblib.dump(model_object, model_path)

    return score

def predict_with_model(model_path, input_data):
    model_object = joblib.load(model_path)

    feature_columns = model_object["feature_columns"]
    preprocessing_method = model_object["preprocessing_method"]
    model = model_object["model"]
    scaler = model_object["scaler"]

    input_df = pd.DataFrame([input_data])
    input_df = convert_numeric_columns(input_df)

    if preprocessing_method == "get_dummies":
        X = pd.get_dummies(input_df[feature_columns], drop_first=True)
        X = X.reindex(columns=model_object["dummy_columns"], fill_value=0)
    else:
        X = apply_ordinal_encoder_for_prediction(
            input_df=input_df,
            encoder=model_object["encoder"],
            categorical_columns=model_object["categorical_columns"],
            numeric_columns=model_object["numeric_columns"]
        )

    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    return prediction.tolist()


def columns_to_json(columns):
    return json.dumps(columns)


def columns_from_json(value):
    return json.loads(value)