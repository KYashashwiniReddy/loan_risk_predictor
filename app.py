from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load saved files
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
encoders = joblib.load("encoders.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/loan/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        education = encoders["Education"].transform(
            [data["Education"]]
        )[0]

        employment = encoders["EmploymentType"].transform(
            [data["EmploymentType"]]
        )[0]

        marital = encoders["MaritalStatus"].transform(
            [data["MaritalStatus"]]
        )[0]

        mortgage = encoders["HasMortgage"].transform(
            [data["HasMortgage"]]
        )[0]

        dependents = encoders["HasDependents"].transform(
            [data["HasDependents"]]
        )[0]

        purpose = encoders["LoanPurpose"].transform(
            [data["LoanPurpose"]]
        )[0]

        cosigner = encoders["HasCoSigner"].transform(
            [data["HasCoSigner"]]
        )[0]

        features = np.array([
            float(data["Age"]),
            float(data["Income"]),
            float(data["LoanAmount"]),
            float(data["CreditScore"]),
            float(data["MonthsEmployed"]),
            float(data["NumCreditLines"]),
            float(data["InterestRate"]),
            float(data["LoanTerm"]),
            float(data["DTIRatio"]),
            education,
            employment,
            marital,
            mortgage,
            dependents,
            purpose,
            cosigner
        ]).reshape(1, -1)

        features = scaler.transform(features)
        features = pca.transform(features)

        prediction = model.predict(features)[0]

        result = (
            "❌ Loan Rejected"
            if prediction == 1
            else "✅ Loan Approved"
        )

        return jsonify({
            "prediction": result
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True
    )