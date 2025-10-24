import torch
import numpy as np

def predict_next_days(model, scaler, scaled_data, seq_length=30, days=10, device='cpu'):
    seq_input = scaled_data[-seq_length:]
    predicted_scaled = []

    for _ in range(days):
        seq_tensor = torch.tensor(seq_input.reshape(1, seq_length, 1), dtype=torch.float32).to(device)
        with torch.no_grad():
            next_scaled = model(seq_tensor).cpu().numpy()[0,0]
        predicted_scaled.append(next_scaled)
        seq_input = np.append(seq_input[1:], next_scaled)
    
    predicted_arr = np.array(predicted_scaled).reshape(-1, 1)
    predicted_prices = scaler.inverse_transform(predicted_arr).flatten()
    return predicted_prices
