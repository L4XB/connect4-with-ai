from agents.alpha_zero_agent.model import ConnectFourNet
from agents.alpha_zero_agent.training import self_play, train
import torch
import os

def main():
    model = ConnectFourNet()
    if os.path.exists("az_model.pth"):
        model.load_state_dict(torch.load("az_model.pth"))
    
    for iteration in range(5):
        print(f"\n=== Iteration {iteration+1} ===")
        data = self_play(model, num_games=200)
        model = train(model, data)
        torch.save(model.state_dict(), "az_model.pth")
        print(f"Model saved")


main()