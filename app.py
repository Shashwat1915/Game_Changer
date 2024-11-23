import io
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, send_file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

app = Flask(__name__)

# Load datasets
datasets = {
    "pl": "pl.csv",
    "uefa": "uefa.csv",
    "world cup": "world_cup.csv"
}

# Global variables
model = None
team_encoder = None
accuracy = None
df = None

# Function to train the model
def train_model(dataset_name):
    global model, team_encoder, accuracy, df
    df = pd.read_csv(datasets[dataset_name])

    # Encode team names
    team_encoder = LabelEncoder()
    df['Team ID'] = team_encoder.fit_transform(df['Team Name'])

    # Define features and target
    features = ['Matches Won', 'Matches Lost', 'Matches Draw', 'Total Points']
    X = df[features]
    y = df['Team ID']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

# Route for the homepage
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle predictions
@app.route("/predict", methods=["POST"])
def predict():
    global df, model, team_encoder, accuracy

    # Get form data
    dataset_name = request.form.get("dataset")
    team1 = request.form.get("team1")
    team2 = request.form.get("team2")

    # Train the model if dataset is chosen
    if dataset_name:
        train_model(dataset_name)

    # Check if teams exist in the dataset
    if team1 not in df['Team Name'].values or team2 not in df['Team Name'].values:
        return render_template("index.html", error="One or both team names are invalid.")

    # Get team stats
    features = ['Matches Won', 'Matches Lost', 'Matches Draw', 'Total Points']
    team1_stats = df[df['Team Name'] == team1][features].values
    team2_stats = df[df['Team Name'] == team2][features].values

    # Predict probabilities for both teams
    team1_prob = model.predict_proba(team1_stats)[0]
    team2_prob = model.predict_proba(team2_stats)[0]

    # Predicted winner
    winner_encoded = model.predict(team1_stats if team1_prob.max() > team2_prob.max() else team2_stats)[0]
    winner = team_encoder.inverse_transform([winner_encoded])[0]

    return render_template(
        "index.html",
        prediction={
            "winner": winner,
            "team1": team1,
            "team1_prob": team1_prob.max() * 100,
            "team2": team2,
            "team2_prob": team2_prob.max() * 100,
            "accuracy": accuracy * 100
        },
        show_chart=True
    )

# Route to generate and display the chart
@app.route("/chart/<team1>/<team2>")
def generate_chart(team1, team2):
    # Generate hypothetical game probabilities for each team
    time = list(range(1, 91, 10))  # Game time in minutes
    team1_chances = np.linspace(np.random.uniform(40, 60), np.random.uniform(60, 80), len(time))
    team2_chances = 100 - team1_chances

    # Plot the probabilities
    plt.figure(figsize=(8, 5))
    plt.plot(time, team1_chances, label=f"{team1} Probability", color="blue", marker="o")
    plt.plot(time, team2_chances, label=f"{team2} Probability", color="red", marker="o")
    plt.fill_between(time, team1_chances, team2_chances, color="gray", alpha=0.1)
    plt.title(f"Game Progression: {team1} vs {team2}")
    plt.xlabel("Game Time (minutes)")
    plt.ylabel("Winning Probability (%)")
    plt.legend()
    plt.grid(True)

    # Save the chart to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True,port=5001)
