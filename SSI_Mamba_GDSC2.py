import torch
import torch.nn as nn
from torch_geometric.nn import HypergraphConv, GCNConv, GINConv, global_max_pool, global_mean_pool, SAGPooling, JumpingKnowledge, global_add_pool
import sys
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
# from torch_scatter import scatter
from mamba_ssm import Mamba

sys.path.append('..')

drug_num = 35
dropout_rate = 0.1
gnn_dim = 128
num_heads = 4


class NodeLevelSAGPooling(nn.Module):
    def __init__(self, in_channels, ratio=1.0, nonlinearity='tanh'):
        super(NodeLevelSAGPooling, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio

        # GCN layer for computing node importance scores
        self.gnn = GCNConv(in_channels, 1)

        # Activation function for scores
        if isinstance(nonlinearity, str):
            self.nonlinearity = getattr(torch, nonlinearity)
        else:
            self.nonlinearity = nonlinearity

    def forward(self, x, edge_index, batch):
        # Compute node scores using GCN
        scores = self.gnn(x, edge_index).squeeze(-1)  # Shape: [num_nodes]
        scores = self.nonlinearity(scores)

        # Normalize scores using softmax (per subgraph)
        normalized_scores = torch.zeros_like(scores)
        for graph_idx in batch.unique():
            mask = batch == graph_idx
            normalized_scores[mask] = torch.softmax(scores[mask], dim=0)

        # Compute weighted features
        x_pooled = global_add_pool(x * normalized_scores.unsqueeze(-1), batch)  # Element-wise multiplication

        return x_pooled, normalized_scores


class DrugGNN(nn.Module):
    def __init__(self, dim_drug, gnn_dim=gnn_dim, dropout_rate=dropout_rate):
        super(DrugGNN, self).__init__()
        # Add normalization layer
        self.drug_normalizer = nn.BatchNorm1d(dim_drug)  # Suitable for global feature normalization

        # ------ drug_layer ------ #
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(dim_drug, gnn_dim * 2),
            nn.ReLU(),
            nn.Linear(gnn_dim * 2, gnn_dim)
        ))  # GCNConv, GraphSAGE, GAT, ChebNet, ResGCN, SAGEConv
        self.batch_conv1 = nn.BatchNorm1d(gnn_dim)  # BatchNorm1d after GINConv

        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(gnn_dim, gnn_dim * 2),
            nn.ReLU(),
            nn.Linear(gnn_dim * 2, gnn_dim)
        ))
        self.batch_conv2 = nn.BatchNorm1d(gnn_dim)

        # SAGPooling with pooling_ratio=1.0 to retain all nodes
        self.pool1 = NodeLevelSAGPooling(gnn_dim, ratio=1.0)
        self.pool2 = NodeLevelSAGPooling(gnn_dim, ratio=1.0)

        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.ReLU()

    def forward(self, drug_feature, drug_adj, ibatch):
        drug_feature = self.drug_normalizer(drug_feature)

        # ------ Drug Training (GIN + SAGPooling) ------ #
        x_drug_local1 = self.conv1(drug_feature, drug_adj)
        x_drug_local1 = self.act(x_drug_local1)
        x_drug_local1 = self.batch_conv1(x_drug_local1)  # BatchNorm
        x_drug_local1 = self.dropout(x_drug_local1)

        x_drug_local2 = self.conv2(x_drug_local1, drug_adj)
        x_drug_local2 = self.act(x_drug_local2)
        x_drug_local2 = self.batch_conv2(x_drug_local2)  # BatchNorm
        x_drug_local2 = self.dropout(x_drug_local2)

        # Node-level SAGPooling for Layer 1
        x_drug_global1, scores1 = self.pool1(x_drug_local1, drug_adj, ibatch)

        # Node-level SAGPooling for Layer 2
        x_drug_global2, scores2 = self.pool2(x_drug_local2, drug_adj, ibatch)

        # Concatenate features from both layers
        x_drug = torch.stack((x_drug_global1, x_drug_global2), dim=1)  # Shape: [num_nodes, 2 * gnn_dim]

        return x_drug


class CellMLP(nn.Module):
    def __init__(self, dim_cellline, output_dim=gnn_dim, dropout_rate=dropout_rate):
        super(CellMLP, self).__init__()
        self.cell_normalizer = nn.BatchNorm1d(dim_cellline)

        # ------ cell line_layer ------ #
        self.fc_cell1 = nn.Linear(dim_cellline, 512)  # 第一层全连接
        self.batch_cell1 = nn.BatchNorm1d(512)       # 第一层 BatchNorm

        self.fc_cell2 = nn.Linear(512, 2 * output_dim)   # 第二层全连接
        self.batch_cell2 = nn.BatchNorm1d(2 * output_dim) # 第二层 BatchNorm

        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.ReLU()                          # 激活函数

    def forward(self, gexpr_data):
        gexpr_data = self.cell_normalizer(gexpr_data)

        # ----cellline_train
        x_cellline1 = self.fc_cell1(gexpr_data)
        x_cellline1 = self.batch_cell1(x_cellline1)      # BatchNorm
        x_cellline1 = self.act(x_cellline1)              # ReLU 激活
        x_cellline1 = self.dropout(x_cellline1)

        x_cellline2 = self.fc_cell2(x_cellline1)
        x_cellline2 = self.batch_cell2(x_cellline2)      # BatchNorm
        x_cellline2 = self.act(x_cellline2)              # ReLU 激活
        x_cellline2 = self.dropout(x_cellline2)

        return x_cellline2


class CrossAttention(nn.Module):
    def __init__(self, input_dim=gnn_dim, heads=num_heads, dropout_rate=dropout_rate):
        super(CrossAttention, self).__init__()
        self.attn1 = nn.MultiheadAttention(embed_dim=input_dim, num_heads=heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim=input_dim, num_heads=heads, batch_first=True)

        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, input_dim)

        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(2)  # After first cross-attention
        self.bn2 = nn.BatchNorm1d(2)  # After second cross-attention

        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.ReLU()

    def forward(self, A, B):
        """
        A: Drug feature matrix A (batch_size, seq_len, input_dim)
        B: Drug or Cell line feature matrix B (batch_size, seq_len, input_dim)
        """
        # Cross-attention (A to B)
        attn_output1, attn_weights1 = self.attn1(A, B, B)
        attn_output1 = attn_output1 + A  # Add residual connection
        attn_output1 = self.bn1(attn_output1)  # Apply BatchNorm
        attn_output1 = self.linear1(attn_output1)
        attn_output1 = self.act(attn_output1)
        attn_output1 = self.dropout(attn_output1)  # Apply Dropout

        # Cross-attention (B to A)
        attn_output2, attn_weights2 = self.attn2(B, A, A)
        attn_output2 = attn_output2 + B  # Add residual connection
        attn_output2 = self.bn2(attn_output2)  # Apply BatchNorm
        attn_output2 = self.linear2(attn_output2)
        attn_output2 = self.act(attn_output2)
        attn_output2 = self.dropout(attn_output2)  # Apply Dropout

        # Concatenate the two outputs
        combined = torch.cat((attn_output1, attn_output2), dim=1)

        return combined


class SSI(nn.Module):
    def __init__(self, input_dim=gnn_dim, heads=num_heads):
        super(SSI, self).__init__()
        self.cross_attention_d1d2 = CrossAttention(input_dim=input_dim, heads=heads)

    def forward(self, x_drug1_local, x_drug2_local):
        """
        x_drug_local1, x_drug_local2: Feature matrices for drug1 and drug2 (batch_size, seq_len, input_dim)
        x_cellline: Cell-line feature matrix (batch_size, seq_len, input_dim)
        """
        # Step 1: Compute attention between drug1 and drug2 (cross-co-attention)
        d1d2 = self.cross_attention_d1d2(x_drug1_local, x_drug2_local)

        return d1d2

class Highway(nn.Module):
    def __init__(self, input_dim=gnn_dim * 18, num_layers=1, dropout_rate=dropout_rate):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.linears = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(input_dim) for _ in range(num_layers)])  # BatchNorm for each layer
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_layers)])  # Dropout for each layer

        # Gate bias initialization to encourage skipping early in training
        for gate in self.gates:
            nn.init.constant_(gate.bias, -1.0)

    def forward(self, x):
        for i in range(self.num_layers):
            # Compute gate and transform
            gate = torch.sigmoid(self.gates[i](x))  # Gate: controls how much to keep
            transform = F.relu(self.linears[i](x))  # Transform: apply non-linearity
            transform = self.bns[i](transform)  # Apply BatchNorm first
            transform = self.dropouts[i](transform)  # Apply Dropout next

            # Highway connection
            x = gate * transform + (1 - gate) * x  # Residual connection controlled by gate
        return x


class ExplanationNet(torch.nn.Module):
    def __init__(self, DrugGNN, CellMLP, SSI, output_dim=gnn_dim, smiles_emb_dim=2, dropout_rate=0.3, vocab_size=209, seq_len=128):
        super(ExplanationNet, self).__init__()
        self.DrugGNN = DrugGNN
        self.CellMLP = CellMLP
        self.SSI = SSI
        self.Highway = Highway(input_dim=output_dim * 12)

        self.bn_cell = nn.BatchNorm1d(2 * gnn_dim)

        self.smiles_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=smiles_emb_dim)
        self.bn_d1_smiles = nn.BatchNorm1d(seq_len)
        self.bn_d2_smiles = nn.BatchNorm1d(seq_len)

        self.mamba = Mamba(
            d_model=smiles_emb_dim,  # Model dimension d_model
            d_state=2,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.bn_smiles_mamba0 = nn.BatchNorm1d(seq_len * smiles_emb_dim * 2)
        self.fc_smiles_mamba1 = nn.Linear(seq_len * smiles_emb_dim * 2, output_dim * 4)
        self.bn_smiles_mamba1 = nn.BatchNorm1d(output_dim * 4)

        self.fc_did2c_smiles1 = nn.Linear(output_dim * 6, output_dim * 6)
        self.bn_did2c_smiles1 = nn.BatchNorm1d(output_dim * 6)


        self.bn_gin_ssi0 = nn.BatchNorm1d(output_dim * 4)
        self.fc_gin_ssi1 = nn.Linear(output_dim * 4, output_dim * 4)
        self.bn_gin_ssi1 = nn.BatchNorm1d(output_dim * 4)

        self.fc_did2c_ssi1 = nn.Linear(output_dim * 6, output_dim * 6)
        self.bn_did2c_ssi1 = nn.BatchNorm1d(output_dim * 6)

        # 定义两层 MLP
        self.bn_cat = nn.BatchNorm1d(output_dim * 12)

        self.fc1 = nn.Linear(output_dim * 12, output_dim * 2)
        self.bn1 = nn.BatchNorm1d(output_dim * 2)

        # 输出层
        self.fc_out = nn.Linear(output_dim * 2, 1)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout for final output
        self.act = nn.ReLU()


    def forward(self, drug_feature, drug_adj, ibatch, gexpr_data, smiles_data, druga_id, drugb_id, cellline_id):
        # Drug and Cell line embeddings
        drug_set = self.DrugGNN(drug_feature, drug_adj, ibatch)
        cell_set = self.CellMLP(gexpr_data)

        # embedding
        d1 = drug_set[druga_id, :]
        d2 = drug_set[drugb_id, :]
        cell_emb = self.bn_cell(cell_set[(cellline_id - drug_num), :])

        # Drug smiles
        d1_smiles = smiles_data[druga_id, :]
        d2_smiles = smiles_data[drugb_id, :]

        d1_smiles = self.smiles_embedding(d1_smiles)
        d2_smiles = self.smiles_embedding(d2_smiles)

        d1_smiles = self.bn_d1_smiles(d1_smiles)
        d2_smiles = self.bn_d2_smiles(d2_smiles)

        d1_smiles = self.mamba(d1_smiles)
        d2_smiles = self.mamba(d2_smiles)

        d1_smiles = d1_smiles.view(d1_smiles.size(0), -1)
        d2_smiles = d2_smiles.view(d2_smiles.size(0), -1)

        d1d2_smiles = torch.cat([d1_smiles, d2_smiles], dim=-1)
        d1d2_smiles = self.bn_smiles_mamba0(d1d2_smiles)
        d1d2_smiles = self.dropout(self.act(self.bn_smiles_mamba1(self.fc_smiles_mamba1(d1d2_smiles))))

        d1d2c_smiles = torch.cat([d1d2_smiles, cell_emb], dim=-1)
        d1d2c_smiles = self.dropout(self.act(self.bn_did2c_smiles1(self.fc_did2c_smiles1(d1d2c_smiles))))

        # Explanation
        #  Call SSI with adjusted inputs
        ssi = self.SSI(d1, d2)

        ssi = ssi.view(ssi.size(0), -1)
        d1d2_gin = self.bn_gin_ssi0(ssi)
        d1d2_gin = self.dropout(self.act(self.bn_gin_ssi1(self.fc_gin_ssi1(d1d2_gin))))

        d1d2c_gin = torch.cat([d1d2_gin, cell_emb], dim=-1)
        d1d2c_gin = self.dropout(self.act(self.bn_did2c_ssi1(self.fc_did2c_ssi1(d1d2c_gin))))

        d1d2c = torch.cat([d1d2c_gin, d1d2c_smiles], dim=-1)

        # Two-layer MLP
        x = self.bn_cat(d1d2c)
        x = self.Highway(x)
        x = self.dropout(self.act(self.bn1(self.fc1(x))))

        # Final output layer with additional dropout
        pred = self.fc_out(x)

        # Squeeze the last dimension
        pred = pred.squeeze(1)

        return pred