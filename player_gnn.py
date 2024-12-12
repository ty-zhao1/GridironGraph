import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import pickle
import scipy.sparse as sp
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import time

class NFLDataset(Dataset):
	def __init__(self, data):
		self.data = data
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		row = self.data.iloc[idx]
		home_players = torch.tensor(row['home_players'], dtype=torch.long)
		away_players = torch.tensor(row['away_players'], dtype=torch.long)
		target = torch.tensor([1.0, 0.0] if row['Home Score'] > row['Away Score'] else [0.0, 1.0], dtype=torch.float)
		return home_players, away_players, target

# 1. Load and preprocess data
print("Loading data")
print("-"*100)
adj_matrix = sp.load_npz('player_adjacency_matrix_sparse.npz')
adj_matrix_dense = adj_matrix.toarray()
game_data = pd.read_csv('nfl-game-info-updated.csv')
team_comp_2005_2023 = pd.read_csv('team_composition_2005_to_2023.csv')

with open('player_encoder.pkl', 'rb') as f:
	player_encoder = pickle.load(f)

def player_to_index(player_name):
	return player_encoder.get(player_name, -1)

# Preprocess team composition data
print("Preprocessing team composition data")
print("-"*100)
team_comp_2005_2023['player_index'] = team_comp_2005_2023['Player Name'].apply(player_to_index)

# Function to get team composition for a specific team and year
def get_team_composition(team, year):
	comp_data = team_comp_2005_2023
	return comp_data[(comp_data['Team'] == team) & (comp_data['Year'] == year)]['player_index'].tolist()

# Preprocess game data to include player indices
print("Preprocessing game data to include player indices")
print("-"*100)
game_data['home_players'] = game_data.apply(lambda row: get_team_composition(row['Home Team'], row['Date']), axis=1)
game_data['away_players'] = game_data.apply(lambda row: get_team_composition(row['Away Team'], row['Date']), axis=1)

# Split data into train (2003-2021) and test (2022-2023)
train_data = game_data[game_data['Date'] < 2022]
train_data = train_data[train_data['Date'] > 2019]
test_data = game_data[game_data['Date'] >= 2022]

# 2. Create graph
print("Creating graph and feature engineering")
print("-"*100)
edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long)
edge_weight = torch.tensor(adj_matrix.data, dtype=torch.float)
num_nodes = adj_matrix.shape[0]

# Feature engineering
player_wins = np.zeros(num_nodes)
player_games = np.zeros(num_nodes)

for _, row in train_data.iterrows():
	home_players = row['home_players']
	away_players = row['away_players']
	home_score = row['Home Score']
	away_score = row['Away Score']
	
	player_games[home_players] += 1
	player_games[away_players] += 1
	
	if home_score > away_score:
		player_wins[home_players] += 1
	elif away_score > home_score:
		player_wins[away_players] += 1

# Calculate sparsity coefficients
sparsity_wins = np.sum(player_wins == 0) / num_nodes
sparsity_games = np.sum(player_games == 0) / num_nodes

print(f"Sparsity coefficient of player_wins: {sparsity_wins:.4f}")
print(f"Sparsity coefficient of player_games: {sparsity_games:.4f}")

# Instead of calculating win percentage, use the raw counts as features
node_features = torch.tensor(np.column_stack((player_wins, player_games)), dtype=torch.float)


# Define GNN model
class GNNModel(torch.nn.Module):
	def __init__(self, num_node_features, hidden_channels):
		super(GNNModel, self).__init__()
		self.conv1 = GCNConv(num_node_features, hidden_channels)
		self.conv2 = GCNConv(hidden_channels, hidden_channels)
		self.linear = torch.nn.Linear(hidden_channels * 2, 2)

	def forward(self, x, edge_index, edge_weight):
		x = self.conv1(x, edge_index, edge_weight).relu()
		x = self.conv2(x, edge_index, edge_weight)
		return x

	def predict(self, x, edge_index, edge_weight, home_players, away_players):
		node_embeddings = self(x, edge_index, edge_weight)
		home_embed = torch.stack([node_embeddings[players].mean(dim=0) for players in home_players])
		away_embed = torch.stack([node_embeddings[players].mean(dim=0) for players in away_players])
		game_embed = torch.cat([home_embed, away_embed], dim=1)
		return self.linear(game_embed)



# Prepare data for PyTorch Geometric
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)
# Create datasets
train_dataset = NFLDataset(train_data)
test_dataset = NFLDataset(test_data)

# Create dataloaders
batch_size = 32  # You can adjust this
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
num_samples = len(train_data)
print(f"Number of training samples: {num_samples}")
num_batches = (num_samples + batch_size - 1) // batch_size  # This formula ensures we account for any remainder
print(f"Number of batches per epoch: {num_batches}")



# Initialize model
model = GNNModel(num_node_features=2, hidden_channels=64)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.BCEWithLogitsLoss()  # Changed to BCEWithLogitsLoss

print("TRAINING LOOP")
print("-"*100)


num_epochs = 10
# for epoch in tqdm(range(num_epochs), desc="Training"):
for epoch in range(num_epochs):
	model.train()
	total_loss = 0
	# batch = 1
	for home_players, away_players, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
		# print(f"Starting batch {batch}/{num_batches}")
		optimizer.zero_grad()
		out = model.predict(data.x, data.edge_index, data.edge_attr, home_players, away_players)
		loss = criterion(out, targets)
		print(loss)
		loss.backward()
		optimizer.step()
		total_loss += loss.item()
		# batch += 1
	print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
	for home_players, away_players, targets in tqdm(test_loader, desc="Evaluating"):
		out = model.predict(data.x, data.edge_index, data.edge_attr, home_players, away_players)
		preds = out.argmax(dim=1)
		correct += (preds == targets.argmax(dim=1)).sum().item()
		total += targets.size(0)

acc = correct / total
print(f'Accuracy: {acc:.4f}')
