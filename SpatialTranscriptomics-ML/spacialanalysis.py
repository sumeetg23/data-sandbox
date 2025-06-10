import scanpy as sc
import squidpy as sq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

class SpatialClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super(SpatialClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)  # up to 10 classes
        )

    def forward(self, x):
        return self.model(x)

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GNNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super(GNNClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
    
def prepare_graph_data(adata):
    # Node features
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    coords = adata.obsm['spatial'] / 1000.0  # Normalize
    X = np.hstack([X, coords])

    # Labels
    y = adata.obs['cluster'].astype('category').cat.codes.values

    # Create edges based on spatial proximity
    from sklearn.neighbors import kneighbors_graph
    adjacency_matrix = kneighbors_graph(coords, n_neighbors=5, mode='connectivity')
    edge_index = np.array(adjacency_matrix.nonzero())

    # Convert to PyTorch Geometric Data
    data = Data(x=torch.tensor(X, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                y=torch.tensor(y, dtype=torch.long))
    return data

def train_gnn(data, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNClassifier(input_dim=data.x.shape[1], num_classes=len(data.y.unique())).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(100):  # Example: 100 epochs
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Save the model
    torch.save(model.state_dict(), os.path.join(output_dir, "gnn_model.pth"))
    return model
def prepare_data(adata, include_coords=True):
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    if include_coords:
        coords = adata.obsm['spatial'] / 1000.0  # Normalize
        X = np.hstack([X, coords])
    y = adata.obs['cluster'].astype('category').cat.codes.values
    return X, y

def train_model(X, y, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    num_classes = len(np.unique(y))
    model = SpatialClassifier(input_dim=X.shape[1], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(50):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).argmax(1).cpu().numpy()
        acc = (preds == y_test.numpy()).mean()
        print(f"Test Accuracy: {acc:.2%}")

    torch.save(model.state_dict(), os.path.join(output_dir, "spatial_model.pt"))
    return model

def plot_predictions(adata, model, X, output_dir):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32)
        preds = model(inputs).argmax(1).numpy()
    adata.obs['prediction'] = preds.astype(str)
    sc.pl.spatial(adata, color='prediction', spot_size=1.3, show=False)
    plt.savefig(os.path.join(output_dir, "spatial_predictions.png"), bbox_inches="tight")
    plt.close()

def evaluate_mlp(model, X, y):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float)
        y_pred = model(X).argmax(dim=1).numpy()
    accuracy = (y_pred == y).mean()
    return accuracy

def evaluate_gnn(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        y_pred = out.argmax(dim=1).cpu().numpy()
    accuracy = (y_pred == data.y.cpu().numpy()).mean()
    return accuracy

def main(args):
    os.makedirs(args.output, exist_ok=True)

    if args.input:
        print(f"Loading dataset from: {args.input}")
        adata = sc.read_h5ad(args.input)
    else:
        print("No input provided. Loading demo dataset from Squidpy...")
        adata = sq.datasets.visium_hne_adata()
        demo_path = os.path.join(args.output, "demo_spatial_data.h5ad")
        adata.write(demo_path)
        print(f"Saved demo dataset to: {demo_path}")

    if 'cluster' not in adata.obs.columns:
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata, resolution=0.5, key_added='cluster')

    # Prepare data for MLP
    X, y = prepare_data(adata, include_coords=args.include_coords)
    mlp_model = train_model(X, y, args.output)

    # Prepare data for GNN
    graph_data = prepare_graph_data(adata)
    gnn_model = train_gnn(graph_data, args.output)

    # Compare performance (example: accuracy)
    print("Evaluating MLP...")
    mlp_accuracy = evaluate_mlp(mlp_model, X, y)
    print(f"MLP Accuracy: {mlp_accuracy}")

    print("Evaluating GNN...")
    gnn_accuracy = evaluate_gnn(gnn_model, graph_data)
    print(f"GNN Accuracy: {gnn_accuracy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatial Transcriptomics Classifier")
    parser.add_argument("--input", type=str, help="Input .h5ad file. If not provided, Squidpy demo dataset is used.")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--include_coords", action="store_true", help="Include spatial coordinates in model input")
    args = parser.parse_args()
    main(args)
