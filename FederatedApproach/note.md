feddiff
Files: feddiff_client.py, feddiff_server.py
Key Feature:
Heterogeneous Federated Learning – Each client can have its own configuration (local epochs, batch size, learning rate, optimizer, model architecture, etc.).
How:
The server (feddiff_server.py) uses a custom HeterogeneousStrategy to send different configs to each client.
The client (feddiff_client.py) reads these configs in its fit() method and adapts its training accordingly.
Example: Client 0 can train for 5 epochs with batch size 256, while Client 1 trains for 1 epoch with batch size 64.
Use Case:
Useful when clients have different compute power, data sizes, network speeds, or need personalized models.
fedsame
Files: fedsame_client.py, fedsame_server.py
Key Feature:
Homogeneous Federated Learning – All clients use the same configuration (same epochs, batch size, learning rate, etc.).
How:
The server (fedsame_server.py) uses a standard FedAvg strategy (or a simple subclass for saving models).
The client (fedsame_client.py) always uses the same training loop and hyperparameters for every client.
Use Case:
Suitable when all clients are similar (same hardware, data, requirements).





 & C:/Users/ASUS/AppData/Local/Microsoft/WindowsApps/python3.11.exe c:/Users/ASUS/Downloads/GUEST/federated/feddiff_server.py
 

& C:/Users/ASUS/AppData/Local/Microsoft/WindowsApps/python3.11.exe C:\Users\ASUS\Downloads\GUEST\federated\fed_diff2.py 


& C:/Users/ASUS/AppData/Local/Microsoft/WindowsApps/python3.11.exe C:\Users\ASUS\Downloads\GUEST\federated\fed_diff2.py --client-id 0 