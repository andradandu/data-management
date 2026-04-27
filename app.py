import os
import uuid
import pandas as pd

from flask import Flask, render_template, redirect, url_for, flash, request
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename

from config import Config
from models import db, User, Dataset, TrainedModel
from forms import RegisterForm, LoginForm
from ml_utils import (
    load_dataset,
    get_shape,
    get_dtypes_html,
    get_head_html,
    get_tail_html,
    get_describe_html,
    drop_empty_rows,
    train_model,
    predict_with_model,
    columns_to_json,
    columns_from_json
)

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"csv"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.cli.command("init-db")
def init_db():
    db.create_all()
    print("Database tables created successfully.")


@app.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        existing_user = User.query.filter(
            (User.email == form.email.data) | (User.username == form.username.data)
        ).first()

        if existing_user:
            flash("Username or email already exists.", "danger")
            return redirect(url_for("register"))

        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)

        db.session.add(user)
        db.session.commit()

        flash("Account created successfully. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html", form=form)


@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()

        if user and user.check_password(form.password.data):
            login_user(user)
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "danger")

    return render_template("login.html", form=form)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    if request.method == "POST":
        file = request.files.get("csv_file")

        if not file or file.filename == "":
            flash("Please choose a CSV file.", "warning")
            return redirect(url_for("dashboard"))

        if not allowed_file(file.filename):
            flash("Only CSV files are allowed.", "danger")
            return redirect(url_for("dashboard"))

        original_filename = secure_filename(file.filename)
        stored_filename = f"user_{current_user.id}_{uuid.uuid4().hex}_{original_filename}"
        path = os.path.join(app.config["UPLOAD_FOLDER"], stored_filename)
        file.save(path)

        try:
            df = pd.read_csv(path)
            rows, columns = get_shape(df)
        except Exception as exc:
            os.remove(path)
            flash(f"Could not read CSV file: {exc}", "danger")
            return redirect(url_for("dashboard"))

        dataset = Dataset(
            user_id=current_user.id,
            original_filename=original_filename,
            stored_filename=stored_filename,
            rows=rows,
            columns=columns
        )
        db.session.add(dataset)
        db.session.commit()

        flash("CSV uploaded successfully.", "success")
        return redirect(url_for("dataset_detail", dataset_id=dataset.id))

    datasets = Dataset.query.filter_by(user_id=current_user.id).order_by(Dataset.uploaded_at.desc()).all()
    trained_models = TrainedModel.query.filter_by(user_id=current_user.id).order_by(TrainedModel.created_at.desc()).all()

    return render_template("dashboard.html", datasets=datasets, trained_models=trained_models)


def get_dataset_or_404(dataset_id):
    return Dataset.query.filter_by(id=dataset_id, user_id=current_user.id).first_or_404()


def get_dataset_path(dataset):
    return os.path.join(app.config["UPLOAD_FOLDER"], dataset.stored_filename)


@app.route("/dataset/<int:dataset_id>")
@login_required
def dataset_detail(dataset_id):
    dataset = get_dataset_or_404(dataset_id)
    df = load_dataset(get_dataset_path(dataset))
    return render_template("dataset.html", dataset=dataset, columns=df.columns.tolist(), table_html=None, title=None)


@app.route("/dataset/<int:dataset_id>/columns")
@login_required
def dataset_columns(dataset_id):
    dataset = get_dataset_or_404(dataset_id)
    df = load_dataset(get_dataset_path(dataset))
    table_html = get_dtypes_html(df)
    return render_template("dataset.html", dataset=dataset, columns=df.columns.tolist(), table_html=table_html, title="Column Names and Details")


@app.route("/dataset/<int:dataset_id>/head", methods=["POST"])
@login_required
def dataset_head(dataset_id):
    dataset = get_dataset_or_404(dataset_id)
    n = int(request.form.get("n", 5))
    df = load_dataset(get_dataset_path(dataset))
    table_html = get_head_html(df, n)
    return render_template("dataset.html", dataset=dataset, columns=df.columns.tolist(), table_html=table_html, title=f"First {n} Rows")


@app.route("/dataset/<int:dataset_id>/tail", methods=["POST"])
@login_required
def dataset_tail(dataset_id):
    dataset = get_dataset_or_404(dataset_id)
    n = int(request.form.get("n", 5))
    df = load_dataset(get_dataset_path(dataset))
    table_html = get_tail_html(df, n)
    return render_template("dataset.html", dataset=dataset, columns=df.columns.tolist(), table_html=table_html, title=f"Last {n} Rows")


@app.route("/dataset/<int:dataset_id>/describe")
@login_required
def dataset_describe(dataset_id):
    dataset = get_dataset_or_404(dataset_id)
    df = load_dataset(get_dataset_path(dataset))
    table_html = get_describe_html(df)
    return render_template("dataset.html", dataset=dataset, columns=df.columns.tolist(), table_html=table_html, title="Basic Statistics")


@app.route("/dataset/<int:dataset_id>/dropna", methods=["POST"])
@login_required
def dataset_dropna(dataset_id):
    dataset = get_dataset_or_404(dataset_id)
    path = get_dataset_path(dataset)

    old_rows, new_rows, new_columns = drop_empty_rows(path)

    dataset.rows = new_rows
    dataset.columns = new_columns
    db.session.commit()

    flash(f"Empty value rows removed. Rows before: {old_rows}, rows after: {new_rows}.", "success")
    return redirect(url_for("dataset_detail", dataset_id=dataset.id))


@app.route("/dataset/<int:dataset_id>/train", methods=["GET", "POST"])
@login_required
def train(dataset_id):
    dataset = get_dataset_or_404(dataset_id)
    df = load_dataset(get_dataset_path(dataset))
    columns = df.columns.tolist()
    if request.method == "POST":
        model_name = request.form.get("model_name", "My Model")
        algorithm_type = request.form.get("algorithm_type")
        algorithm_name = request.form.get("algorithm_name")
        preprocessing_method = request.form.get("preprocessing_method")
        target_column = request.form.get("target_column")
        feature_columns = request.form.getlist("feature_columns")

        if not feature_columns:
            flash("Please select at least one feature column.", "warning")
            return redirect(url_for("train", dataset_id=dataset.id))

        if algorithm_type in ["regression", "classification"] and not target_column:
            flash("Please select a target column.", "warning")
            return redirect(url_for("train", dataset_id=dataset.id))

        if target_column in feature_columns:
            flash("Target column cannot also be a feature column.", "warning")
            return redirect(url_for("train", dataset_id=dataset.id))

        model_filename = f"model_user_{current_user.id}_{uuid.uuid4().hex}.joblib"
        model_path = os.path.join(app.config["MODEL_FOLDER"], model_filename)

        try:
            score = train_model(
                df=df,
                feature_columns=feature_columns,
                target_column=target_column,
                algorithm_type=algorithm_type,
                algorithm_name=algorithm_name,
                preprocessing_method=preprocessing_method,
                model_path=model_path
            )
        except Exception as exc:
            flash(f"Model training failed: {exc}", "danger")
            return redirect(url_for("train", dataset_id=dataset.id))

        trained_model = TrainedModel(
            user_id=current_user.id,
            dataset_id=dataset.id,
            name=model_name,
            algorithm_type=algorithm_type,
            algorithm_name=algorithm_name,
            target_column=target_column,
            feature_columns=columns_to_json(feature_columns),
            preprocessing_method=preprocessing_method,
            model_path=model_filename
        )

        db.session.add(trained_model)
        db.session.commit()

        if score is not None:
            flash(f"Model trained successfully. Score: {score:.4f}", "success")
        else:
            flash("Clustering model trained successfully.", "success")

        return redirect(url_for("predict", model_id=trained_model.id))

    return render_template("train.html", dataset=dataset, columns=columns)


@app.route("/predict/<int:model_id>", methods=["GET", "POST"])
@login_required
def predict(model_id):
    trained_model = TrainedModel.query.filter_by(id=model_id, user_id=current_user.id).first_or_404()
    feature_columns = columns_from_json(trained_model.feature_columns)
    prediction = None

    if request.method == "POST":
        input_data = {}
        for column in feature_columns:
            input_data[column] = request.form.get(column)

        model_path = os.path.join(app.config["MODEL_FOLDER"], trained_model.model_path)

        try:
            prediction = predict_with_model(model_path, input_data)
        except Exception as exc:
            flash(f"Prediction failed: {exc}", "danger")

    return render_template("predict.html", trained_model=trained_model, feature_columns=feature_columns, prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)