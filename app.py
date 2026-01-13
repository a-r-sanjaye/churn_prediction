from flask import Flask, render_template, request
from predict import predict_churn

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Pass data as a dictionary instead of a raw list of values
        # This allows predict.py to map values to the correct features
        input_data = request.form.to_dict()
        pred, prob = predict_churn(input_data)
        
        return render_template(
            "result.html",
            prediction="CHURN" if pred == 1 else "NO CHURN",
            probability=round(prob * 100, 2)
        )
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
