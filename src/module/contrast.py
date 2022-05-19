import torch
import torch.nn as nn

def InfoNCE(sim_matrix):
    pos_matrix = torch.eye(sim_matrix.shape[0]).to(sim_matrix.device)

    sim_matrix_norm = sim_matrix / (torch.sum(sim_matrix, dim=1).view(-1, 1) + 1e-8)
    loss = -torch.log((sim_matrix_norm * pos_matrix).sum(dim=-1)).mean()

    return loss


class Contrast(nn.Module):
    def __init__(self, h):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(h.hidden_dim, h.hidden_dim),
            nn.ELU(),
            nn.Linear(h.hidden_dim, h.hidden_dim)
        )
        self.tau = h.tau
        self.lam = h.lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, d):
        z_proj1 = self.proj(d.z1)
        z_proj2 = self.proj(d.z2)
        matrix_1to2 = self.sim(z_proj1, z_proj2)
        matrix_2to1 = matrix_1to2.t()

        lori_1to2 = InfoNCE(matrix_1to2)
        lori_2to1 = InfoNCE(matrix_2to1)

        loss = self.lam * lori_1to2 + (1 - self.lam) * lori_2to1
        return loss
