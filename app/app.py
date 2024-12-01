from dash import Dash, dcc, html, Input, Output, State, callback
import plotly.express as px
import numpy as np
import pickle
import dash
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
# Load the model
filename = './model/group4V2.model'
fileUnsup = './model/model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

# Load model unsup
def load_model():
    try:
        return pickle.load(open(fileUnsup, 'rb'))
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

model_data = load_model()
if model_data:
    model = model_data['model']
    preprocessor = model_data['preprocessor']


# Initialize the Dash app
app = Dash(__name__)
app.layout = html.Div(
    style={'backgroundColor': '#198f51', 'fontFamily': 'Arial', 'textAlign': 'center', 'minHeight': '100vh', 'display': 'flex', 'flexDirection': 'column'},
    children=[
        # Navbar
        html.Div(
            style={'display': 'flex', 'justifyContent': 'space-between', 'padding': '10px', 'backgroundColor': '#f5f5f5'},
            children=[
                html.Div('Group 4', style={'fontSize': '20px', 'fontWeight': 'bold', 'marginLeft': '10px'}),
                html.Div('Bank Customer Churn Prediction', style={'fontSize': '30px', 'fontWeight': 'bold', 'textAlign': 'center', 'flex': '1','marginLeft': '300px'}),
                html.Div(
                    children=[
                        html.Button('Home Prediction', style={'marginRight': '10px', 'backgroundColor': '#d9534f', 'color': 'white', 'height': '30px', 'borderRadius': '5px'})
                    ]
                )
            ]
        ),
 
        # Feature Importance Display
        # Feature Importance Display with Buttons
        # Main Div for Feature Importance and Clustering
        html.Div(
            style={'padding': '20px', 'backgroundColor': '#ffffff', 'margin': '20px auto', 'width': '80%', 'borderRadius': '5px'},
            children=[
                # Buttons for switching between views
                html.Div(
                    style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'},
                    children=[
                        html.Button('Feature Importance', id='feature-importance-button', style={'marginRight': '10px', 'backgroundColor': '#007bff', 'color': 'white', 'padding': '10px 20px', 'borderRadius': '5px', 'fontSize': '16px'}),
                        html.Button('Clustering Results', id='clustering-results-button', style={'backgroundColor': '#FF1493', 'color': 'white', 'padding': '10px 20px', 'borderRadius': '5px', 'fontSize': '16px'})
                    ]
                ),
                # Dynamic content area to switch between graph and image
                html.Div(id='dynamic-content', style={'textAlign': 'center'})
            ]
        ),

        
         # Main content area
        html.Div(
            style={'display': 'flex', 'justifyContent': 'center', 'padding': '20px'},
            children=[
                # Left Side - Feature Inputs with Orange Background
                html.Div(
                    style={
                        'padding': '30px',
                        'backgroundColor': '#ffffff',
                        'borderRadius': '15px',
                        'width': '60%',  # Expanded width
                        'marginRight': '20px',
                        'boxShadow': '0px 4px 8px rgba(0, 0, 0, 0.1)',
                        'textAlign': 'left'
                    },
                    children=[
                        html.H3("Input Features", style={'color': '#333333', 'textAlign': 'center', 'marginBottom': '20px'}),
                        
                        # Left Column
                        html.Div(
                            style={'display': 'flex', 'justifyContent': 'space-between'},
                            children=[
                                # Column 1
                                html.Div(
                                    style={'width': '48%'},
                                    children=[
                                        html.Label("Age", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                                        dcc.Input(
                                            id='age',
                                            type="number",
                                            placeholder="Enter your age",
                                            min = 10,
                                            max = 99,
                                            style={
                                                'width': '100%',
                                                'padding': '10px',
                                                'borderRadius': '10px',
                                                'border': 'none',
                                                'backgroundColor': '#ffcc66',
                                                'color': '#616161',
                                                'boxShadow': '0px 2px 4px rgba(0, 0, 0, 0.1)',
                                                'fontSize': '16px'
                                            }
                                        ),
                                        html.Label("Job", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px', 'marginTop': '15px'}),
                                        dcc.Dropdown(
                                            id='job',
                                            options=[
                                                {'label': 'admin', 'value': 0},
                                                {'label': 'technician', 'value': 9},
                                                {'label': 'services', 'value': 7},
                                                {'label': 'management', 'value': 4},
                                                {'label': 'retired', 'value': 5},
                                                {'label': 'blue-collar', 'value': 1},
                                                {'label': 'unemployed', 'value': 10},
                                                {'label': 'entrepreneur', 'value': 2},
                                                {'label': 'housemaid', 'value': 3},
                                                {'label': 'unknown', 'value': 11},
                                                {'label': 'self-employed', 'value': 6},
                                                {'label': 'student', 'value': 8}
                                            ],
                                            placeholder="Select Job",
                                            style={
                                                'marginBottom': '15px',
                                                'borderRadius': '10px',
                                                'backgroundColor': '#ffcc66',
                                                'color': '#616161',
                                                'boxShadow': '0px 2px 4px rgba(0, 0, 0, 0.1)',
                                            }
                                        ),
                                        html.Label("Marital", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                                        dcc.Dropdown(
                                            id='marital',
                                            options=[
                                                {'label': 'married', 'value': 1},
                                                {'label': 'single', 'value': 2},
                                                {'label': 'divorced', 'value': 0}
                                            ],
                                            placeholder="Select marital status",
                                            style={
                                                'marginBottom': '15px',
                                                'borderRadius': '10px',
                                                'backgroundColor': '#ffcc66',
                                                'color': '#616161',
                                                'boxShadow': '0px 2px 4px rgba(0, 0, 0, 0.1)',
                                            }
                                        ),
                                        html.Label("Education", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                                        dcc.Dropdown(
                                            id="education",
                                            options=[
                                                {'label': 'secondary', 'value': 1},
                                                {'label': 'tertiary', 'value': 2},
                                                {'label': 'primary', 'value': 0},
                                                {'label': 'unknown', 'value': 3}
                                            ],
                                            placeholder="Select education level",
                                            style={
                                                'marginBottom': '15px',
                                                'borderRadius': '10px',
                                                'backgroundColor': '#ffcc66',
                                                'color': '#616161',
                                                'boxShadow': '0px 2px 4px rgba(0, 0, 0, 0.1)',
                                            }
                                        )
                                    ]
                                ),

                                # Column 2
                                html.Div(
                                    style={'width': '48%'},
                                    children=[
                                        html.Label("Default (Has credit in default?)", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                                        dcc.Dropdown(
                                            id="default",
                                            options=[
                                                {'label': 'yes', 'value': 1},
                                                {'label': 'no', 'value': 0}
                                            ],
                                            placeholder="Select Default",
                                            style={
                                                'marginBottom': '15px',
                                                'borderRadius': '10px',
                                                'backgroundColor': '#ffcc66',
                                                'color': '#616161',
                                                'boxShadow': '0px 2px 4px rgba(0, 0, 0, 0.1)',
                                            }
                                        ),
                                        html.Label("Balance (Yearly balance)", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                                        dcc.Input(
                                            id="balance",
                                            type="number",
                                            placeholder="Enter balance",
                                            min = 1,
                                            style={
                                                'width': '100%',
                                                'padding': '10px',
                                                'borderRadius': '10px',
                                                'border': 'none',
                                                'backgroundColor': '#ffcc66',
                                                'color': '#616161',
                                                'boxShadow': '0px 2px 4px rgba(0, 0, 0, 0.1)',
                                                'fontSize': '16px'
                                            }
                                        ),
                                        html.Label("Housing (Has housing loan?)", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                                        dcc.Dropdown(
                                            id="housing",
                                            options=[
                                                {'label': 'yes', 'value': 1},
                                                {'label': 'no', 'value': 0}
                                            ],
                                            placeholder="Select housing",
                                            style={
                                                'marginBottom': '15px',
                                                'borderRadius': '10px',
                                                'backgroundColor': '#ffcc66',
                                                'color': '#616161',
                                                'boxShadow': '0px 2px 4px rgba(0, 0, 0, 0.1)',
                                            }
                                        ),
                                        html.Label("Loan (Has personal loan?)", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                                        dcc.Dropdown(
                                            id="loan",
                                            options=[
                                                {'label': 'yes', 'value': 1},
                                                {'label': 'no', 'value': 0}
                                            ],
                                            placeholder="Select loan",
                                            style={
                                                'marginBottom': '15px',
                                                'borderRadius': '10px',
                                                'backgroundColor': '#ffcc66',
                                                'color': '#616161',
                                                'boxShadow': '0px 2px 4px rgba(0, 0, 0, 0.1)',
                                            }
                                        ),
                                        html.Label("Deposit", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                                        dcc.Dropdown(
                                            id="deposit",
                                            options=[
                                                {'label': 'yes', 'value': 1},
                                                {'label': 'no', 'value': 0}
                                            ],
                                            placeholder="Select deposit",
                                            style={
                                                'marginBottom': '15px',
                                                'borderRadius': '10px',
                                                'backgroundColor': '#ffcc66',
                                                'color': '#616161',
                                                'boxShadow': '0px 2px 4px rgba(0, 0, 0, 0.1)',
                                            }
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                ),
                
                # Right Side - Prediction Result
                html.Div(
                    style={'padding': '20px', 'backgroundColor': '#99ccff', 'borderRadius': '25px', 'width': '30%', 'position': 'relative'},
                    children=[
                        html.Div("RandomForestClassifier & K-Means Clustering model", style={'position': 'absolute', 'top': '-20px', 'right': '-20px', 'backgroundColor': '#f24822', 'padding': '10px', 'borderRadius': '5px', 'fontWeight': 'bold', 'color': 'white'}),
                        html.H1("Result", style={'color': '#333333'}),
                        html.P(id='prediction-output', children="Please enter all information to get prediction", style={'fontSize': '24px', 'color': '#C70039', 'fontWeight': 'bold'})
                    ]
                )
            ]
        ),
        
        # Prediction Button
        html.Div(
            style={'marginTop': '20px'},
            id='predict-button', 
            n_clicks=0,
            children=[
                html.Button('Prediction', style={'backgroundColor': '#000000', 'color': '#ffffff', 'width': '700px', 'height': '50px', 'borderRadius': '5px', 'fontWeight': 'bold'})
            ]
        ),
        
        # Footer
        html.Div(
            '',
            style={'padding': '20px', 'backgroundColor': '#f5f5f5', 'marginTop': '20px'}
        )
    ]
)

@app.callback(
    Output('dynamic-content', 'children'),
    [Input('feature-importance-button', 'n_clicks'),
     Input('clustering-results-button', 'n_clicks')]
)

def update_content(feature_clicks, clustering_clicks):
    # Default display when no button is clicked
    if not feature_clicks and not clustering_clicks:
        return html.H3("Select an option to display content.")

    ctx = dash.callback_context  # Get the triggered context
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Display Feature Importance Graph
        if button_id == 'feature-importance-button':
            return dcc.Graph(
                id='feature-importance-graph',
                figure=px.bar(
                    x=['Age', 'Job', 'Marital', 'Education', 'Default', 'Balance', 'Housing', 'Loan', 'Deposit'],
                    y=[0.1939, 0.0981, 0.0255, 0.0353, 0.0046, 0.5029, 0.0711, 0.0389, 0.0298],
                    labels={'x': "Features", 'y': "Importance Score"}
                )
            )

        # Display Clustering Image
        elif button_id == 'clustering-results-button':
            return html.Img(
                src='/assets/scatter_plot.png',  # Ensure the image is in the `assets` folder
                style={'width': '80%', 'height': 'auto', 'borderRadius': '10px'}
            )

    # Default case (should not hit here ideally)
    return html.H3("Select an option to display content.")

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('age', 'value'),
    State('job', 'value'),
    State('marital', 'value'),
    State('education', 'value'),
    State('default', 'value'),
    State('balance', 'value'),
    State('housing', 'value'),
    State('loan', 'value'),
    State('deposit', 'value')
)

def predict_churn(n_clicks, age, job, marital, education, default, balance, housing, loan, deposit):
    if n_clicks > 0:
        # Validate inputs
        if age is None or age <= 0:
            return "Please enter an age greater than 0."
        if balance is None or balance <= 0:
            return "Please enter a balance greater than 0."

        # Ensure all inputs are provided
        if all(v is not None for v in [age, job, marital, education, default, balance, housing, loan, deposit]):
            try:
                # Prepare raw input data
                sample = [age, job, marital, education, default, balance, housing, loan, deposit]

                # Load and apply MinMaxScaler
                with open('./model/scaler.pkl', 'rb') as file:
                    scaler = pickle.load(file)
                print("Scaler loaded successfully:", scaler)

                # Scale balance
                balance_reshaped = np.array([[sample[5]]])  # Reshape balance to 2D
                scaled_balance = scaler.transform(balance_reshaped)[0][0]  # Scale balance
                sample[5] = scaled_balance  # Update scaled value

                print("Sample after scaling:", sample)
                sample2 = np.array([sample])

                # Make churn prediction
                prediction = loaded_model.predict(sample2)
                churn_probabilities = loaded_model.predict_proba(sample2)[:, 1]
                data_prob = churn_probabilities[0] * 100

                # Process results
                if data_prob > 50:
                    recommendation = generate_recommendations(
                        age=age, job=job, marital=marital, education=education, 
                        default=default, balance=balance, housing=housing, 
                        loan=loan, deposit=deposit
                    )
                    final_result = html.Div(
                        children=[
                            html.Span(f"{data_prob:.2f}% will be Churn Customer (Churn customer)",
                                      style={'color': '#C70039', 'font-weight': 'bold'}),
                            html.Br(),
                            html.H4("Recommendations:", style={'marginTop': '15px'}),
                            recommendation
                        ]
                    )
                else:
                    final_result = html.Div(
                        children=[
                            html.Span(f"{data_prob:.2f}% will be Churn Customer (Not churn customer)",
                                      style={'color': '#008000', 'font-weight': 'bold'})
                        ]
                    )
                return final_result

            except Exception as e:
                return f"An error occurred during prediction: {str(e)}"

        else:
            return "Please fill out all fields."

    return "Click the button to predict."


def generate_recommendations(age, job, marital, education, default, balance, housing, loan, deposit):
    """
    Generate cluster-based recommendations using the K-Means model.
    """
    try:
        # Define mapping options
        JOB_OPTIONS = [
            {'label': 'admin', 'value': 0},
            {'label': 'technician', 'value': 9},
            {'label': 'services', 'value': 7},
            {'label': 'management', 'value': 4},
            {'label': 'retired', 'value': 5},
            {'label': 'blue-collar', 'value': 1},
            {'label': 'unemployed', 'value': 10},
            {'label': 'entrepreneur', 'value': 2},
            {'label': 'housemaid', 'value': 3},
            {'label': 'unknown', 'value': 11},
            {'label': 'self-employed', 'value': 6},
            {'label': 'student', 'value': 8}
        ]
        MARITAL_OPTIONS = [
            {'label': 'married', 'value': 1},
            {'label': 'single', 'value': 2},
            {'label': 'divorced', 'value': 0}
        ]
        EDUCATION_OPTIONS = [
            {'label': 'secondary', 'value': 1},
            {'label': 'tertiary', 'value': 2},
            {'label': 'primary', 'value': 0},
            {'label': 'unknown', 'value': 3}
        ]
        BINARY_OPTIONS = [
            {'label': 'yes', 'value': 1},
            {'label': 'no', 'value': 0}
        ]

        # Map numeric values back to their labels
        job_label = next(opt['label'] for opt in JOB_OPTIONS if opt['value'] == job)
        marital_label = next(opt['label'] for opt in MARITAL_OPTIONS if opt['value'] == marital)
        education_label = next(opt['label'] for opt in EDUCATION_OPTIONS if opt['value'] == education)
        default_label = next(opt['label'] for opt in BINARY_OPTIONS if opt['value'] == default)
        housing_label = next(opt['label'] for opt in BINARY_OPTIONS if opt['value'] == housing)
        loan_label = next(opt['label'] for opt in BINARY_OPTIONS if opt['value'] == loan)
        deposit_label = next(opt['label'] for opt in BINARY_OPTIONS if opt['value'] == deposit)

        # Create input dataframe with human-readable labels
        input_data = pd.DataFrame({
            'age': [age],
            'job': [job_label],
            'marital': [marital_label],
            'education': [education_label],
            'default': [default_label],
            'balance': [balance],
            'housing': [housing_label],
            'loan': [loan_label],
            'deposit': [deposit_label]
        })

        print("Input data for recommendation:", input_data)

        # Make prediction using K-Means model
        transformed_data = preprocessor.transform(input_data)
        prediction = model.predict(transformed_data)

        if prediction[0] == 0:  # High-value customers
            promotion = "Exclusive perks: 1-year fee waiver on premium account."
        elif prediction[0] == 1:  # At-risk customers
            promotion = "10% cashback on debit card transactions for 3 months."
        elif prediction[0] == 2:  # Low-value customers
            promotion = "$20 bonus for maintaining 500 Baht minimum balance for 6 months."
        elif prediction[0] == 3:  # New customers
            promotion = "Welcome bonus: 50 Baht for setting up direct deposit within 30 days."
        else:
            promotion = "No promotion available"
        # Determine promotion based on cluster
        # promotion_map = {
        #     0: "5%",
        #     1: "10%",
        #     2: "15%",
        #     3: "20%"
        # }
        # promotion = promotion_map.get(prediction[0], "No promotion available")

        # Return result as HTML
        return html.Div([
            html.H4("Cluster Assignment"),
            html.P(f"Result: Cluster {prediction[0]}"),
            html.H4("Promotion Details"),
            html.Div(
                f"Customer will get the {promotion} promotion",
                style={
                    'padding': '10px',
                    'backgroundColor': '#4CAF50',
                    'color': 'white',
                    'borderRadius': '4px',
                    'marginTop': '10px'
                }
            )
        ])

    except Exception as e:
        return html.Div(f"An error occurred during recommendation generation: {str(e)}",
                        style={'color': 'red'})


# Run the app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
