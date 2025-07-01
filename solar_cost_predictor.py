import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load and preprocess the dataset
df = pd.read_csv("solar_installation_data.csv")
df = pd.get_dummies(df, columns=["panel_type", "location_type"], drop_first=True)

# Features and target
X = df.drop("installation_cost", axis=1)
y = df["installation_cost"]

# Train model
model = LinearRegression()
model.fit(X, y)

# User inputs
print("\n Welcome to Solar Installation Cost Estimator\n")
panel_type = input("Enter panel type (mono / poly / thin-film): ").strip().lower()
battery_backup = int(input("Battery backup required? (1 = Yes, 0 = No): "))
installation_area = float(input("Enter installation area (sq. ft.): "))
location_type = input("Enter location type (rooftop / ground): ").strip().lower()
subsidy = int(input("Enter government subsidy amount (₹): "))
labour_cost = int(input("Enter labour cost per panel (₹): "))
wattage = int(input("Enter wattage per panel (e.g., 320, 350, 400): "))

# Auto-calculate number of panels
sqft_per_panel = 17.5  # average area occupied per panel
number_of_panels = int(installation_area // sqft_per_panel)
print(f"\n Estimated number of panels based on area: {number_of_panels}")

# Create input dictionary with required format
input_data = {
    "number_of_panels": number_of_panels,
    "battery_backup_required": battery_backup,
    "installation_area_sqft": installation_area,
    "government_subsidy": subsidy,
    "labour_cost_per_panel": labour_cost,
    "wattage_per_panel": wattage,
    "panel_type_poly": 1 if panel_type == "poly" else 0,
    "panel_type_thin-film": 1 if panel_type == "thin-film" else 0,
    "location_type_rooftop": 1 if location_type == "rooftop" else 0
}

# Ensure all expected columns are present
for col in X.columns:
    if col not in input_data:
        input_data[col] = 0

# Predict
input_df = pd.DataFrame([input_data])
predicted_cost = model.predict(input_df)[0]

# Output
print(f"\n Estimated Installation Cost: ₹{predicted_cost:,.2f}")
