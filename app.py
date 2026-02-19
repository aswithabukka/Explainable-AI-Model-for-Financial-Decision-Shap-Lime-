from flask import Flask, request, render_template, jsonify, redirect, url_for
import pandas as pd
import pickle
import shap

app = Flask(__name__)

with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('shap_explainer.pkl', 'rb') as f:
    shap_explainer = pickle.load(f)

# Define numerical and categorical columns
numerical_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# Encoding to original mapping for feature interpretation
encoded_to_original_mapping = {
    'person_home_ownership_RENT': 'person_home_ownership',
    'person_home_ownership_OWN': 'person_home_ownership',
    'person_home_ownership_MORTGAGE': 'person_home_ownership',
    'person_home_ownership_OTHER': 'person_home_ownership',

    'loan_intent_EDUCATION': 'loan_intent',
    'loan_intent_PERSONAL': 'loan_intent',
    'loan_intent_HOMEIMPROVEMENT': 'loan_intent',
    'loan_intent_MEDICAL': 'loan_intent',
    'loan_intent_DEBTCONSOLIDATION': 'loan_intent',
    'loan_intent_VENTURE': 'loan_intent',

    'loan_grade_A': 'loan_grade',
    'loan_grade_B': 'loan_grade',
    'loan_grade_C': 'loan_grade',
    'loan_grade_D': 'loan_grade',
    'loan_grade_E': 'loan_grade',
    'loan_grade_F': 'loan_grade'
}

# Function to map UI inputs to dummy variables
# Function to map UI inputs to dummy variables with True/False values
def map_ui_to_dummies(input_data):
    """
    Converts UI input features into dummy-encoded columns using True/False values, consistent with training.
    """
    # Initialize the dummy-encoded dictionary
    encoded_data = {}

    # Manual encoding for 'person_home_ownership'
    ownership_mapping = ['OTHER', 'OWN', 'RENT']
    for value in ownership_mapping:
        encoded_data[f'person_home_ownership_{value}'] = input_data['person_home_ownership'] == value

    # Manual encoding for 'loan_intent'
    loan_intent_mapping = ['EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE']
    for value in loan_intent_mapping:
        encoded_data[f'loan_intent_{value}'] = input_data['loan_intent'] == value

    # Manual encoding for 'loan_grade'
    loan_grade_mapping = ['B', 'C', 'D', 'E', 'F', 'G']
    for value in loan_grade_mapping:
        encoded_data[f'loan_grade_{value}'] = input_data['loan_grade'] == value

    # Manual encoding for 'cb_person_default_on_file'
    encoded_data['cb_person_default_on_file_Y'] = input_data['cb_person_default_on_file'] == 'Y'

    # Include the original numerical columns as is
    for key, value in input_data.items():
        if key not in ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']:
            encoded_data[key] = value

    return encoded_data


@app.route('/', methods=['GET'])
def home():
    # Render index.html which should contain the form for user input
    return render_template('index.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    # Parse input data from form
    input_data = request.form.to_dict()
    
    # Convert numerical inputs to float
    for col in numerical_cols:
        input_data[col] = float(input_data[col])

    # Map inputs to dummy variables
    encoded_input = map_ui_to_dummies(input_data)

    # Convert encoded data to DataFrame
    input_df = pd.DataFrame([encoded_input])
    input_df = input_df[rf_model.feature_names_in_]

    # Preprocessing Step 1: Standardize numerical features
    input_df[numerical_cols] = input_df[numerical_cols].astype(float)
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    print(rf_model.feature_names_in_)
    print(input_df.columns)


    # Make a prediction
    prediction = rf_model.predict(input_df)[0]
    prediction_proba = rf_model.predict_proba(input_df)[0][1] if hasattr(rf_model, 'predict_proba') else None
    explanation_1 = generate_loan_decision_explanation(input_df, rf_model, shap_explainer)
    # Send results to frontend
    result = {
        'prediction': 'Approved' if prediction == 1 else 'Rejected',
        'explanation': explanation_1
    }
    print(input_df)
    return render_template('results.html', result=result)
    

def get_suggestion_for_improvement(feature, input_data):
    """
    Provide a detailed suggestion for improving a specific feature based on the original input value.
    """
    suggestions = {
        'person_age': "Your age indicates your financial maturity. Lenders often look for a stable financial history as you grow older. Focus on building a consistent credit record and demonstrating responsible financial behavior.",
        
        'person_income': "Your income is a key factor in assessing your ability to repay the loan. Consider increasing your income by exploring higher-paying job opportunities, negotiating a raise, or adding additional sources of income.",
        
        'person_emp_length': "Lenders value job stability when evaluating creditworthiness. Try to maintain consistent employment, ideally for 2 years or more, as it demonstrates reliability and financial stability.",
        
        'loan_amnt': "The loan amount you are requesting appears high compared to your financial profile. Consider requesting a smaller loan amount or increasing your income to reduce the financial risk associated with this loan.",
        
        'loan_int_rate': "The interest rate on your loan indicates higher risk to the lender. Improve your credit score or shop around for lenders offering better rates. Demonstrating better financial stability can also help lower this rate.",
        
        'loan_percent_income': "The percentage of your income that the loan represents is too high. Reduce the loan amount or work to increase your income, so the loan becomes a smaller portion of your financial obligations.",
        
        'cb_person_cred_hist_length': "Your credit history length impacts your perceived reliability. Build a longer credit history by responsibly using credit over time and ensuring you have no missed payments.",
        
        'cb_person_default_on_file_Y':  "A past default on your credit file raises red flags for lenders. Focus on clearing any outstanding defaults, repaying your debts on time, and rebuilding your credit profile through consistent financial discipline.",
        
        # Home ownership categories
        'person_home_ownership_RENT': "Renting can indicate less financial stability than owning a home. Consider transitioning to home ownership, which demonstrates greater financial commitment and stability.",
        'person_home_ownership_MORTGAGE': "Having a mortgage is a positive sign, but timely payments are crucial. Ensure you maintain regular payments to improve your creditworthiness.",
        'person_home_ownership_OWN': "Owning your home outright is a strong indicator of financial stability. Continue managing property-related expenses responsibly to maintain your profile.",
        'person_home_ownership_OTHER': "Unusual or less common home ownership types can be perceived as higher risk. Ensure stable living arrangements and demonstrate consistency in managing housing-related finances.",
        
        # Loan intent categories
        'loan_intent_EDUCATION': "Loans for education are viewed positively if they improve your earning potential. Provide a clear plan showing how the loan will help achieve higher income or better career prospects.",
        'loan_intent_PERSONAL': "Personal loans can indicate higher financial risk. Demonstrate responsible financial planning and ensure the loan has a specific, well-justified purpose.",
        'loan_intent_HOMEIMPROVEMENT': "Loans for home improvement are often seen as investments. Highlight how the loan will increase property value or livability to reassure lenders of its utility.",
        'loan_intent_MEDICAL': "Medical loans are necessary but can indicate financial strain. Show evidence of stable financial planning for medical expenses to reassure lenders of your repayment ability.",
        'loan_intent_DEBTCONSOLIDATION': "Debt consolidation loans can simplify debt management but require a clear repayment plan. Show how consolidating debt will improve your financial situation and enable easier repayment.",
        'loan_intent_VENTURE': "Loans for ventures can be risky. Provide a strong, detailed business plan, including projected returns, to build confidence in your ability to repay the loan.",
        
        # Loan grades
        'loan_grade_A': "Your loan grade indicates low risk. Maintain your current financial habits to secure favorable terms in future loans.",
        'loan_grade_B': "Your loan grade is relatively low risk but could improve. Focus on enhancing your credit score and reducing any outstanding debts to reach Grade A.",
        'loan_grade_C': "Your loan grade indicates moderate risk. Reduce outstanding debts, improve your credit score, and demonstrate consistent financial discipline to improve your grade.",
        'loan_grade_D': "Your loan grade shows higher risk. Focus on increasing your income, reducing your debt-to-income ratio, and improving your overall financial stability.",
        'loan_grade_E': "Your loan grade reflects significant risk. Address key financial issues by repaying debts, increasing income, and rebuilding your creditworthiness over time.",
        'loan_grade_F': "Your loan grade indicates the highest risk level. Take immediate steps to address defaults, reduce debt, and demonstrate responsible financial management to improve your credit profile."
    }
    # Map encoded feature back to original value
    if feature in encoded_to_original_mapping:
        original_feature = encoded_to_original_mapping[feature]
        original_value = input_data.get(original_feature, None)
        if isinstance(suggestions.get(feature), dict):
            return suggestions[feature].get(original_value, "Work on improving this factor to enhance your loan eligibility.")
    return suggestions.get(feature, "Work on improving this factor to enhance your loan eligibility.")

def create_prompt(status, positive_factors, negative_factors, feature_values,input_data_df):
    feature_descriptions = {
        'person_age': "Age of the individual applying for the loan.",
        'person_income': "Annual income of the individual.",
        'person_emp_length': "Employment length of the individual in years.",
        'loan_amnt': "The loan amount requested by the individual.",
        'loan_int_rate': "The interest rate associated with the loan.",
        'loan_percent_income': "The percentage of income represented by the loan amount.",
        'cb_person_cred_hist_length': "The length of credit history for the individual.",
        'person_home_ownership_OTHER': "The individual has an unusual or less common home ownership type.",
        'person_home_ownership_OWN': "The individual owns their home outright.",
        'person_home_ownership_RENT': "The individual is currently renting a property.",
        'loan_intent_EDUCATION': "The loan is intended for educational purposes.",
        'loan_intent_HOMEIMPROVEMENT': "The loan is intended for home improvement.",
        'loan_intent_MEDICAL': "The loan is intended for medical purposes.",
        'loan_intent_PERSONAL': "The loan is intended for personal use.",
        'loan_intent_VENTURE': "The loan is intended for a venture or business purpose.",
        'loan_grade_B': "The loan grade indicates relatively low risk, but not as creditworthy as Grade A.",
        'loan_grade_C': "The loan grade indicates moderate creditworthiness.",
        'loan_grade_D': "The loan grade indicates higher risk compared to previous grades.",
        'loan_grade_E': "The loan grade indicates lower creditworthiness and higher risk.",
        'loan_grade_F': "The loan grade indicates significant credit risk.",
        'loan_grade_G': "The loan grade indicates the highest credit risk.",
        'cb_person_default_on_file_Y': "The individual has a history of defaults on their credit file."
    }

    prompt = f"The loan application has been **{status}**. Below is the detailed analysis:\n\n"
    if status == "rejected":
        prompt += "**Reasons for Rejection:**\n"
        for idx, (feature, impact) in enumerate(negative_factors.items(), start=1):
            description = feature_descriptions.get(feature, "Description not available.")
            value = feature_values.loc[0, feature] if feature in feature_values else "Value not available"
            prompt += f"{idx}. **{feature}**: This factor negatively impacted your application.\n"
            prompt += f"   - Feature description: {description}\n"

    if status == "approved":
        prompt += "**Positive Factors:**\n"
        for idx, (feature, impact) in enumerate(positive_factors.items(), start=1):
            description = feature_descriptions.get(feature, "Description not available.")
            value = feature_values.loc[0, feature] if feature in feature_values else "Value not available"
            prompt += f"{idx}. **{feature}**: This factor positively influenced your application.\n"
            prompt += f"   - Feature description: {description}\n"

    if status == "rejected":
        prompt += "\n"
        prompt += "**Suggestions for Improvement:**\n"
        for idx, (feature, _) in enumerate(negative_factors.items(), start=1):
            prompt += f"{idx}. {get_suggestion_for_improvement(feature,input_data_df)}\n"

    return prompt

def generate_loan_decision_explanation(user_data, rf_model, explainer):
    """
    Function to generate a loan decision explanation based on user's data with impact and improvement suggestions.
    """
    user_data_df = pd.DataFrame(user_data, index=[0])
    
    # Make the prediction
    y_pred = rf_model.predict(user_data_df)
    
    # SHAP explanation
    shap_values = explainer.shap_values(user_data_df)

    # Determine which class SHAP values to use
# Determine which class SHAP values to use
    predicted_class = y_pred[0]  # either 0 or 1 for binary classification
# Select the array for the predicted class, and then the first (and only) sample’s values
    shap_values_user = shap_values[0, :, predicted_class]

    
    # Sort features by impact (SHAP value)
    feature_impact = pd.Series(shap_values_user, index=user_data_df.columns).sort_values(ascending=False)
    
    # Separate top positive and negative impacts
    positive_factors = feature_impact[feature_impact > 0].head(3)  # Top 3 positive factors
    negative_factors = feature_impact[feature_impact < 0].tail(3)  # Top 3 negative factors
    
    # Generate explanation
    status = "approved" if y_pred[0] == 1 else "rejected"
    return create_prompt(status, positive_factors, negative_factors, user_data_df, user_data)


if __name__ == '__main__':
    app.run(debug=True)
