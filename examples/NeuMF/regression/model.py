import torch
from torch import nn


class GMF(nn.Module):
    def __init__(self, num_user, num_item, mf_dim=10, trainable=True):
        super().__init__()
        self.trainable = trainable
        self.mf_user_emb = nn.Embedding(num_embeddings=num_user, embedding_dim=mf_dim)
        self.mf_item_emb = nn.Embedding(num_embeddings=num_item, embedding_dim=mf_dim)
        if trainable:  # 预训练
            self.linear = nn.Sequential(
                nn.Linear(mf_dim, 1),
                #nn.Sigmoid()
            )
        else:
            trained = torch.load('weights/GMF.pt').state_dict()
            for name, val in self.named_parameters():
                val.data = trained[name]
                val.requires_grad = False

    def forward(self, user_id, item_id):
        mf_vec = self.mf_user_emb(user_id) * self.mf_item_emb(item_id)
        if self.trainable:
            pred = self.linear(mf_vec)
            return pred.squeeze()
        else:
            return mf_vec


class MLP(nn.Module):
    def __init__(self, num_user, num_item, mlp_layers=None, trainable=True):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [10]
        self.trainable = trainable
        self.mlp_user_emb = nn.Embedding(num_embeddings=num_user, embedding_dim=mlp_layers[0] // 2)
        self.mlp_item_emb = nn.Embedding(num_embeddings=num_item, embedding_dim=mlp_layers[0] // 2)
        #print(self.mlp_user_emb)
       
        self.mlp = nn.ModuleList()
        for i in range(1, len(mlp_layers)):
            self.mlp.append(nn.Linear(mlp_layers[i - 1], mlp_layers[i]))
            self.mlp.append(nn.ReLU())
        if trainable:
            self.linear = nn.Sequential(
                nn.Linear(mlp_layers[-1], 1),
                #nn.Sigmoid()
            )
        else:
            trained = torch.load('weights/MLP.pt').state_dict()
            for name, val in self.named_parameters():
                val.data = trained[name]
                val.requires_grad = False

    def forward(self, user_id, item_id):
        # print(self.mlp_item_emb.num_embeddings)

        # print(item_id.min())
        # print(item_id.max())

        # print(self.mlp_user_emb(user_id).size())
        # print(self.mlp_item_emb(item_id).size())

        mlp_vec = torch.cat([self.mlp_user_emb(user_id), self.mlp_item_emb(item_id)], dim=-1)
        for layer in self.mlp:
            mlp_vec = layer(mlp_vec)
        if self.trainable:
            prediction = self.linear(mlp_vec)
            return prediction.squeeze()
        else:
            return mlp_vec


class NeuMF(nn.Module):

    def __init__(self, num_user, num_item, mf_dim=10, mlp_layers=None, use_pretrain=True):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [10]
        self.gmf = GMF(num_user, num_item, mf_dim, trainable=not use_pretrain)  # 默认直接使用预训练好的权重
        self.mlp = MLP(num_user, num_item, mlp_layers=mlp_layers, trainable=not use_pretrain)
        self.linear = nn.Sequential(
            nn.Linear(mf_dim + mlp_layers[-1], 1),
            #nn.Sigmoid()
        )

    def forward(self, user_id, item_id):
        gmf_vec = self.gmf(user_id, item_id)
        mlp_vec = self.mlp(user_id, item_id)
        # NueMF
        cat = torch.cat([gmf_vec, mlp_vec], dim=-1)
        prediction = self.linear(cat)
        return prediction.squeeze()
