# House Price Prediction

A machine learning project that estimates house prices (in ₹ Lakhs) from property details such as location, size, BHK, furnishing, and amenities. It uses a Decision Tree Regressor and includes a Streamlit web app for making predictions.

## Project Structure

```
.
├── E06_house price data less.csv    # Dataset (999 properties)
├── E06_House Price Prediction.ipynb # EDA, preprocessing, training, evaluation
├── app.py                           # Streamlit web app
├── house_price_model.pkl            # Trained Decision Tree Regressor
├── vectorizer.pkl                   # Fitted DictVectorizer
├── encoder.pkl                      # Fitted OrdinalEncoder
├── features.pkl                     # Ordered list of 20 input features
└── README.md
```

## Dataset

`E06_house price data less.csv` has 999 rows and 23 columns, with no missing values. It covers 20 Indian states and 42 cities.

| Column | Description |
|---|---|
| `State`, `City`, `Locality` | Location (`Locality` is a generic ID and is dropped) |
| `Property_Type` | Apartment / Independent House / Villa |
| `BHK` | Number of bedrooms |
| `Size_in_SqFt` | Built-up area (502 – 5,000) |
| `Price_in_Lakhs` | **Target**: price in ₹ Lakhs (11.59 – 499.57) |
| `Price_per_SqFt` | Price per square foot (0.00 – 0.97, in Lakhs) |
| `Year_Built`, `Age_of_Property` | Construction year and age |
| `Furnished_Status` | Unfurnished / Semi-furnished / Furnished |
| `Floor_No`, `Total_Floors` | Floor details |
| `Nearby_Schools`, `Nearby_Hospitals` | Counts of nearby facilities |
| `Public_Transport_Accessibility` | Low / Medium / High |
| `Parking_Space`, `Security` | Yes / No |
| `Amenities` | Comma-separated list (Gym, Pool, Garden, etc.) |
| `Facing` | North / South / East / West |
| `Owner_Type` | Owner / Builder / Broker |
| `Availability_Status` | Ready_to_Move / Under_Construction |

`ID` is also dropped, so the model uses 20 input features.

## Methodology

1. **Drop** `ID` and `Locality`.
2. **Ordinal encoding** (`OrdinalEncoder` with explicit category order) for `Property_Type`, `Furnished_Status`, `Public_Transport_Accessibility`, `Facing`, `Security`.
3. **Lowercase** all remaining text columns.
4. **Sampling**: a random 40% of the data (400 rows) is used to speed up training.
5. **Split**: 80% train / 20% test (320 / 80 rows, `random_state=42`).
6. **DictVectorizer** one-hot encodes the remaining categorical columns (`State`, `City`, `Parking_Space`, `Amenities`, `Owner_Type`, `Availability_Status`).
7. **Models**: Linear Regression and `DecisionTreeRegressor(max_depth=10, random_state=42)`.

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 118.77 | 148.11 | 0.031 |
| **Decision Tree (max_depth=10)** | **35.21** | **47.06** | **0.902** |

The Decision Tree is the saved model.

## Web App

`app.py` is a Streamlit app where you enter state, city, property type, BHK, size, price per sq.ft, year built, furnishing, floors, nearby schools/hospitals, transport, parking, security, amenities, facing, owner type, and availability. It returns an estimated price in ₹ Lakhs and shows the raw input data in an expander.

## Getting Started

### Prerequisites

- Python 3.9+

### Installation

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn jupyter
```

> Use the scikit-learn version the model was trained with to avoid pickle compatibility warnings.

### Run the app

```bash
streamlit run app.py
```

Keep all four `.pkl` files in the same folder as `app.py`.

### Retrain the model

Run all cells in the notebook; it regenerates the four `.pkl` files.

## Known Issues & Suggested Improvements

- **Price per Sq.Ft unit mismatch in the app.** In the dataset, `Price_per_SqFt` is in Lakhs (about 0.1, e.g. 489.76 ÷ 4740 ≈ 0.10), but the app asks for values from 1,000 upward (default 5,000). Those inputs are far outside the training range and will produce unreliable predictions.
- **Possible target leakage.** `Price_per_SqFt` looks like `Price_in_Lakhs ÷ Size_in_SqFt`, so it encodes the answer the model is trying to predict. The R² of 0.90 is likely inflated, and a user cannot know this value before the price is known. Try removing it and re-evaluating.
- **Small training set.** Only 400 of 999 rows are used, and evaluation is on just 80 rows. Use the full dataset and cross-validation.
- **Free-text inputs.** City, Amenities, Owner Type, and Availability Status are text boxes. The model expects exact lowercase values (e.g. `owner`, `builder`, `broker`, `ready_to_move`, `under_construction`), and unseen values are silently ignored. Replace them with dropdowns built from the training data.
- **Limited State list.** The app offers 5 states, while the model was trained on 20.
- **Notebook bug.** One evaluation cell prints metrics from the Linear Regression predictions instead of the Decision Tree's. The saved model's real scores are the ones in the table above.
- **No `requirements.txt`.** Add one with pinned versions.
- Try Random Forest or Gradient Boosting and tune hyperparameters with `GridSearchCV`.

## Tech Stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Streamlit# House-Price-Prediction
