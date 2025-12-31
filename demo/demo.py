from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
preprocess = joblib.load(os.path.join(BASE_DIR, "preprocess.pkl"))

models = {
    "Random Forest": joblib.load(os.path.join(BASE_DIR, "random_forest_model.pkl")),
    "XGBoost": joblib.load(os.path.join(BASE_DIR, "xgboost_model.pkl")),
    "Linear Regression": joblib.load(os.path.join(BASE_DIR, "linear_regression_model.pkl")),
    "Gradient Boosting": joblib.load(os.path.join(BASE_DIR, "gradient_boosting_model.pkl")),
}

FEATURE_NAMES = [
    "Giờ học mỗi ngày",
    "Giờ sử dụng mạng xã hội",
    "Làm thêm",
    "Tỷ lệ điểm danh (%)",
    "Giờ ngủ",
    "Chất lượng chế độ ăn",
    "Tần suất tập thể dục",
    "Đánh giá sức khỏe tinh thần",
    "GPA học kỳ trước",
    "Mức độ căng thẳng",
    "Nguy cơ bỏ học",
    "Thời gian dùng màn hình",
    "Môi trường học tập",
    "Tiếp cận lớp phụ đạo / gia sư",
    "Mức độ hỗ trợ của cha mẹ",
    "Mức độ động lực",
    "Mức độ lo âu khi thi",
    "Phong cách học tập",
    "Điểm quản lý thời gian",
]

CAT_COLS = [
    "Làm thêm",
    "Chất lượng chế độ ăn",
    "Tần suất tập thể dục",
    "Đánh giá sức khỏe tinh thần",
    "Mức độ căng thẳng",
    "Nguy cơ bỏ học",
    "Môi trường học tập",
    "Tiếp cận lớp phụ đạo / gia sư",
    "Mức độ hỗ trợ của cha mẹ",
    "Mức độ động lực",
    "Mức độ lo âu khi thi",
    "Phong cách học tập",
    "Điểm quản lý thời gian",
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    selected_model = None
    error = None
    form_data = {}

    if request.method == "POST":
        form_data = request.form.to_dict()

        try:
            selected_model = request.form.get("model")
            if selected_model not in models:
                raise ValueError("❌ Chưa chọn mô hình")

            model = models[selected_model]

            input_data = {}
            for feature in FEATURE_NAMES:
                value = request.form.get(feature)

                if value is None or value.strip() == "":
                    raise ValueError(f"❌ Thiếu giá trị cho biến: <b>{feature}</b>")

                input_data[feature] = value

            df_input = pd.DataFrame([input_data])

            df_input = df_input.apply(pd.to_numeric, errors="coerce")

            for col in CAT_COLS:
                df_input[col] = df_input[col].astype(str)
            if df_input.isna().any().any():
                nan_cols = df_input.columns[df_input.isna().any()].tolist()
                raise ValueError(
                    f"❌ Dữ liệu không hợp lệ ở cột: <b>{', '.join(nan_cols)}</b>"
                )

            print("📥 INPUT DATA (FINAL):")
            print(df_input)
            print("📌 DTYPE:")
            print(df_input.dtypes)

            X_processed = preprocess.transform(df_input)

            prediction = float(model.predict(X_processed)[0])
            print("🎯 PREDICTION:", prediction)

        except Exception as e:
            error = str(e)
            print("❌ LỖI:", error)

    return render_template(
        "index.html",
        prediction=prediction,
        selected_model=selected_model,
        error=error,
        form_data=form_data,
    )


if __name__ == "__main__":
    app.run(debug=True)
