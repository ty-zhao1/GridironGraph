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
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

class NFLDataset(Dataset):
	def __init__(self, data, max_players=100):  # Set max_players to a reasonable value
		self.data = data
		self.max_players = max_players  # Maximum number of players per team

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		row = self.data.iloc[idx]
		home_players = torch.tensor(row['home_players'], dtype=torch.long)
		away_players = torch.tensor(row['away_players'], dtype=torch.long)
		
		# Pad home players and away players to max_players
		home_players_padded = torch.cat([home_players, torch.zeros(self.max_players - len(home_players), dtype=torch.long)])
		away_players_padded = torch.cat([away_players, torch.zeros(self.max_players - len(away_players), dtype=torch.long)])
		
		target = torch.tensor([1.0, 0.0] if row['Home Score'] > row['Away Score'] else [0.0, 1.0], dtype=torch.float)
		
		return home_players_padded[:self.max_players], away_players_padded[:self.max_players], target
	
	def print_sample(self, idx):
		"""Prints the sample at the given index in a readable format."""
		row = self.data.iloc[idx]
		home_players = row['home_players']
		away_players = row['away_players']
		target = [1.0, 0.0] if row['Home Score'] > row['Away Score'] else [0.0, 1.0]
		
		print(f"Sample Index: {idx}")
		print(f"Home Players: {home_players}")
		print(f"Away Players: {away_players}")
		print(f"Target: {target}")

# Custom collate function (if needed)
def collate_fn(batch):
	home_players_batch = pad_sequence([item[0] for item in batch], batch_first=True)
	away_players_batch = pad_sequence([item[1] for item in batch], batch_first=True)
	targets_batch = torch.stack([item[2] for item in batch])
	return home_players_batch, away_players_batch, targets_batch

# 1. Load and preprocess data
print("Loading data")
print("-"*100)
game_data = pd.read_csv('nfl-game-info-updated.csv')

# Modify 'Home Team' and 'Away Team' columns: make uppercase and remove spaces and hyphens
game_data['Home Team'] = game_data['Home Team'].str.upper().str.replace(' ', '').str.replace('-', '')
game_data['Away Team'] = game_data['Away Team'].str.upper().str.replace(' ', '').str.replace('-', '')
print(game_data.head())
team_comp_2005_2023 = pd.read_csv('team_composition_2005_to_2023.csv')
team_comp_2005_2023['Team'] = team_comp_2005_2023['Team'].str.upper().str.replace(' ', '').str.replace('-', '')
print(team_comp_2005_2023.head())

team_to_players = defaultdict(lambda: defaultdict(list))
for _, row in team_comp_2005_2023.iterrows():
	team = row['Team']
	year = row['Year']
	player = row['Player Name']
	team_to_players[team][year].append(player)

all_players = set()
for team in team_to_players.values():
	for players in team.values():
		all_players.update(players)

print(len(all_players))

player_encoder = {player: idx for idx, player in enumerate(sorted(all_players))} # USE IN FUTURE PARTS
adj_matrix = defaultdict(lambda: defaultdict(int))

def player_to_index(player):
	return player_encoder[player]

# Preprocess team composition data
print("Preprocessing team composition data")
print("-"*100)
team_comp_2005_2023['player_index'] = team_comp_2005_2023['Player Name'].apply(player_to_index)
print(team_comp_2005_2023.head())

# Function to get team composition for a specific team and year
def get_team_composition(team, year):
	comp_data = team_comp_2005_2023
	return comp_data[(comp_data['Team'] == team) & (comp_data['Year'] == year)]['player_index'].tolist()

# Preprocess game data to include player indices
print("Preprocessing game data to include player indices")
print("-"*100)
print(game_data.head())
game_data['home_players'] = game_data.apply(lambda row: get_team_composition(row['Home Team'], row['Date']), axis=1)
game_data['away_players'] = game_data.apply(lambda row: get_team_composition(row['Away Team'], row['Date']), axis=1)

train_data = game_data[game_data['Date'] < 2020]
test_data = game_data[game_data['Date']  < 2022]
test_data = test_data[test_data['Date']  >= 2020]
train_data = train_data[train_data['Date'] > 2017]
total_data = game_data[game_data['Date'] < 2022]
total_data = total_data[total_data['Date'] > 2017]
print(train_data.head())

print("Computing adjacency matrix")
print("-"*100)
for _, game in tqdm(total_data.iterrows(), desc="Computing adjacency matrix"):
	winner = game['Winning Team']
	loser = game['Losing Team']
	nfl_year = game['Date']
	
	if winner in team_to_players and nfl_year in team_to_players[winner]:
		winning_players = team_to_players[winner][nfl_year]
	else:
		continue

	if loser in team_to_players and nfl_year in team_to_players[loser]:
		losing_players = team_to_players[loser][nfl_year]
	else:
		continue

	for winner_player in winning_players:
		winner_idx = player_encoder[winner_player]
		for loser_player in losing_players:
			loser_idx = player_encoder[loser_player]
			adj_matrix[winner_idx][loser_idx] += 1

adj_matrix_df = pd.DataFrame.from_dict({player: dict(edges) for player, edges in adj_matrix.items()}).fillna(0)
print(adj_matrix_df)
num_players = len(all_players)
print("Transform into numpy array")
print("-"*100)
adj_np = np.zeros((num_players, num_players))
for i in tqdm(range(num_players), desc = "Transform into numpy array"):
	for j in range(num_players):
		adj_np[i][j] = adj_matrix[i][j]

# 2. Create graph
print("Creating graph and feature engineering")
print("-"*100)
edge_index = torch.tensor(np.array(adj_np.nonzero()), dtype=torch.long)
edge_weight = torch.tensor(adj_np[adj_np.nonzero()].astype(float), dtype=torch.float)
num_nodes = adj_np.shape[0]
print(f"Number of nodes is: {num_nodes}")

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
test_dataset.print_sample(2)

# Create dataloaders
batch_size = 32  # You can adjust this
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
num_samples = len(train_data)
print(f"Number of training samples: {num_samples}")
num_batches = (num_samples + batch_size - 1) // batch_size  # This formula ensures we account for any remainder
print(f"Number of batches per epoch: {num_batches}")



# Initialize model
model = GNNModel(num_node_features=2, hidden_channels=64)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()  # Changed to BCEWithLogitsLoss

print("TRAINING LOOP")
print("-"*100)


# Training loop
num_epochs = 10
for epoch in range(num_epochs):
	model.train()
	total_loss = 0
	correct_predictions = 0
	total_samples = 0
	
	for home_players, away_players, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
		optimizer.zero_grad()
		out = model.predict(data.x, data.edge_index, data.edge_attr, home_players, away_players)
		loss = criterion(out, targets)
		print(f"Loss: {loss.item()}")
		loss.backward()
		optimizer.step()
		
		total_loss += loss.item()
		preds = torch.sigmoid(out)
		preds_labels = (preds > 0.5).float() 
		if targets.size(1) > 1:
			targets_labels = targets.argmax(dim=1)
			preds_labels = preds_labels.argmax(dim=1)
		else:
			targets_labels = targets
		
		correct_predictions += (preds_labels == targets_labels).sum().item()
		total_samples += targets.size(0)

	avg_loss = total_loss / len(train_loader)
	train_accuracy = correct_predictions / total_samples
	print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

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
