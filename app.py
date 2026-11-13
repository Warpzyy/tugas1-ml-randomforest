from flask import Flask, jsonify, render_template, request
from rf import run_forest

app = Flask(__name__)

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/forest")
def forest():
    try:
        return jsonify(run_forest(request.get_json()))
    except Exception as e:  # input tidak valid
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
