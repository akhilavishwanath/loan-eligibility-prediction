import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import gradio as gr

# 🔹 Load dataset
df = pd.read_csv("loan.csv")

# 🔹 Handle missing values
df = df.ffill()

# 🔹 Encode categorical data
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# 🔹 Features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# 🔹 Train model
model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

# 🔹 Prediction function
def predict_loan(ApplicantIncome, LoanAmount):
    input_data = np.zeros((1, X.shape[1]))

    input_data[0][X.columns.get_loc("ApplicantIncome")] = ApplicantIncome
    input_data[0][X.columns.get_loc("LoanAmount")] = LoanAmount

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        return "Loan Approved ✅"
    else:
        return "Loan Rejected ❌"


# 💎 PREMIUM BLACK + GOLD CSS (WITH FOOTER REMOVED)
css = """
body {
    background: #0b0b0b;
    color: #FFD700;
}

.gradio-container {
    background: #0b0b0b !important;
}

/* Headings and labels */
h1, h2, h3, label {
    color: #FFD700 !important;
}

/* Input fields */
input, textarea {
    background-color: #1a1a1a !important;
    color: #FFD700 !important;
    border: 1px solid #FFD700 !important;
}

/* Buttons */
button {
    background: linear-gradient(135deg, #FFD700, #b89600) !important;
    color: black !important;
    font-weight: bold;
    border-radius: 8px !important;
}

/* Output box */
textarea {
    background-color: #1a1a1a !important;
    color: #FFD700 !important;
    border: 1px solid #FFD700 !important;
}

/* ❌ REMOVE FOOTER */
footer {
    display: none !important;
}
"""

# 🔹 UI Layout
with gr.Blocks(css=css) as app:

    gr.Markdown("# 💼 Loan Eligibility Predictor")
    gr.Markdown("### AI-Based Smart Loan Approval System")

    with gr.Row():
        income = gr.Number(label="💰 Applicant Income")
        loan = gr.Number(label="🏦 Loan Amount")

    btn = gr.Button("🚀 Check Eligibility")
    output = gr.Textbox(label="📊 Result")

    btn.click(predict_loan, inputs=[income, loan], outputs=output)

# 🔹 Launch for Hugging Face
app.launch(server_name="0.0.0.0", server_port=7860)