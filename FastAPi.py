import pandas as pd

# Load Excel file
file_path = "EasyJet-complete.xlsx"
xls = pd.ExcelFile(file_path)

# Read a specific sheet
df = pd.read_excel(xls, sheet_name="Financials")  
print(df.head())  # Display first rows
df = pd.read_excel("EasyJet-complete.xlsx")
df.to_csv("EasyJet-financials.csv", index=False)  # Auto-convert on the backend
from fastapi import FastAPI
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime, timedelta
import os

app = FastAPI()

# File Paths
EXCEL_FILE = "EasyJet-complete.xlsx"
CSV_FILE = "EasyJet-financials.csv"

# Convert Excel to CSV if it doesn't exist
def convert_excel_to_csv():
    if not os.path.exists(CSV_FILE):
        df = pd.read_excel(EXCEL_FILE, sheet_name="Financials")
        df.to_csv(CSV_FILE, index=False)
        print("Converted Excel to CSV.")

convert_excel_to_csv()

# Load financial data
financials = pd.read_csv(CSV_FILE)

# Load historical stock data (Mock data, replace with real data source)
stocks = pd.DataFrame({
    "date": pd.date_range(start="2023-01-01", periods=365, freq='D'),
    "price": np.cumsum(np.random.randn(365)) + 50
})

# PyTorch LSTM model for stock price prediction
class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(StockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.fc(h[:, -1, :])

# Initialize and train model
model = StockPredictor(input_size=1, hidden_size=50, output_size=1, num_layers=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

def train_model():
    X_train = torch.tensor(stocks['price'].values[:-30], dtype=torch.float32).reshape(-1, 30, 1)
    y_train = torch.tensor(stocks['price'].values[30:], dtype=torch.float32).reshape(-1, 1)
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
    print("Model training complete!")

def predict_stock_prices():
    past_prices = torch.tensor(stocks['price'].values[-30:], dtype=torch.float32).reshape(1, 30, 1)
    model.eval()
    with torch.no_grad():
        predicted_prices = model(past_prices).numpy().flatten()
    return [{"date": (datetime.today() + timedelta(days=i)).strftime('%Y-%m-%d'), "predicted_price": float(predicted_prices[i-1])} for i in range(1, 31)]

@app.get("/api/financials")
def get_financials():
    return financials.to_dict(orient="records")

@app.get("/api/predictions")
def get_predictions():
    return predict_stock_prices()

@app.get("/api/ratios")
def get_ratios():
    return [{"name": "P/E Ratio", "value": pe} for pe in financials['pe_ratio']]

# Train model on startup
train_model()
