import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the GCN model with embeddings and output layer for binary classification
class GCNEmbeddingModel(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, embedding_dim):
		super(GCNEmbeddingModel, self).__init__()
		self.conv1 = GCNConv(input_dim, hidden_dim)
		self.conv2 = GCNConv(hidden_dim, embedding_dim)
		self.fc = torch.nn.Linear(embedding_dim, 1)  # Output for 1 class (binary classification)

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = self.conv2(x, edge_index)  # The output of this layer will be the node embeddings
		x = self.fc(x)  # Apply final fully connected layer to get logits for 1 class (binary)
		return x

# Load data and process as before...

# Example: Generate binary labels for win/loss classification
labels = np.zeros(len(teams))  # 0 for loss, 1 for win
for _, row in game_info_2024.iterrows():
	home_team_idx = team_to_idx[row['Home Team']]
	away_team_idx = team_to_idx[row['Away Team']]
	
	if row["Home Score"] > row["Away Score"]:
		labels[home_team_idx] = 1  # Home team wins
	elif row["Away Score"] > row["Home Score"]:
		labels[away_team_idx] = 1  # Away team wins

# Save labels as .npy
np.save('labels.npy', labels)

# Convert labels to tensor
y_train_tensor = torch.tensor(labels, dtype=torch.float).view(-1, 1)  # For BCEWithLogitsLoss, shape (batch_size, 1)

# Initialize model, criterion, optimizer
input_dim = node_features.size(1)
hidden_dim = 16
embedding_dim = 16

model = GCNEmbeddingModel(input_dim=input_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
criterion = torch.nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(20):
	model.train()
	
	# Forward pass
	outputs = model(data)  # Logits for each node (team)
	
	# Compute loss
	loss = criterion(outputs, y_train_tensor)
	
	# Backward pass and optimization
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	
	print(f"Epoch {epoch+1}/20, Loss: {loss.item()}")

# Further steps (e.g., evaluation, saving models) would follow...
# Evaluate the model
model.eval()  # Set the model to evaluation mode (disables dropout, batch norm, etc.)

# Forward pass for evaluation
with torch.no_grad():  # No need to compute gradients during evaluation
	outputs = model(data)
	predictions = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
	predictions = (predictions > 0.5).float()  # Convert to 0 or 1 based on a threshold of 0.5

# Compute accuracy
accuracy = (predictions.squeeze() == y_train_tensor.squeeze()).float().mean()
print(f"Accuracy: {accuracy.item() * 100:.2f}%")

# Save the trained model
torch.save(model.state_dict(), 'gcn_embedding_model.pth')
print("Model saved to 'gcn_embedding_model.pth'")

# Load the trained model (if saved using state_dict)
model = GCNEmbeddingModel(input_dim=input_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
model.load_state_dict(torch.load('gcn_embedding_model.pth'))
model.eval()  # Set to evaluation mode

# Inference example (assuming you have a new data object for inference)
with torch.no_grad():
	outputs = model(data)
	predictions = torch.sigmoid(outputs)
	predictions = (predictions > 0.5).float()
	print(predictions)

# Collect losses during training for plotting
losses = []

for epoch in range(20):
	model.train()
	outputs = model(data)
	loss = criterion(outputs, y_train_tensor)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	# Store loss for visualization
	losses.append(loss.item())
	print(f"Epoch {epoch+1}/20, Loss: {loss.item()}")

# Plot the training loss over epochs
plt.plot(range(1, 21), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score
	
	# Convert outputs and labels to numpy arrays for scikit-learn functions
predictions = predictions.squeeze().numpy()
labels = y_train_tensor.squeeze().numpy()

precision = precision_score(labels, predictions)
recall = recall_score(labels, predictions)
f1 = f1_score(labels, predictions)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
