



import torch
import torch.nn as nn
import torch.nn.functional as F
from gcl import GCRL
from torch.nn.functional import pairwise_distance

class SynCo(nn.Module):
    def __init__(self, in_channels, num_classes, config,args):
        super(SynCo, self).__init__()

        self.config = config
       
        self.encoder = GCRL(in_channels, config)
        indim = config['cl_hdim']
        hdim = config['hidden_channels'] 
        self.T = config['tau'] 
        self.lam_r = config['lam_r']
        self.dist_net = SimNet(indim, shared_proj=self.encoder.disc.proj)
        self.classifier = nn.Linear(indim, num_classes)
        self.c = None
       
        self.sampler_type = args.sampler_type
        
        if self.sampler_type == 'vae':
            self.sampler = VAESampler(input_dim=indim, hidden_dim=hdim, latent_dim=hdim//2)
        elif self.sampler_type == 'diffusion':
            self.sampler = DiffusionSampler(input_dim=indim, hidden_dim=hdim, timesteps=config.get('diffusion_steps', 50))
        else:
            raise ValueError(f"error sampler_type: {self.sampler_type}")

    

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.dist_net.reset_parameters()
        self.classifier.reset_parameters()
        self.sampler.reset_parameters()

    
    def sample_centroid(self, h, tau):
        initial_center = h.mean(dim=0)   
        distances = torch.sum((h - initial_center.unsqueeze(0)) ** 2, dim=1)
        weights = torch.softmax(-distances / tau, dim=0)
        final_center = (h * weights.view(-1, 1)).sum(dim=0)
        return final_center

    def forward(self, dataset):
        z_pos, z_neg = self.encoder(dataset)  # (n, cl_hdim), (n, cl_hdim)
        self.c = self.sample_centroid(z_pos.detach(), self.T)  # (cl_hdim)
        dist = self.dist_net(z_pos.detach(), self.c).squeeze()  
        return z_pos, z_neg, dist

    def classify(self, dataset, node_idx):
        z = self.encoder.embed(dataset)  
        logits = self.classifier(z[node_idx].detach())
        return logits

    @torch.no_grad()
    def detect(self, dataset, node_idx):
        z = self.encoder.embed(dataset)  
        alpha = 1
        neg_dist = -self.dist_net(z[node_idx], self.c).squeeze()  
        return alpha * neg_dist
    
    

    
    @torch.no_grad()
    def get_pseudo_ood_scores(self, num_samples, device):
        z_sample = self.sample(sample_size=num_samples, device=device)
        alpha = 1
        neg_dist = -self.dist_net(z_sample, self.c).squeeze()
        return alpha * neg_dist


    def loss_compute(self, dataset_ind, dataset_ood=None, reduction='mean'):
        train_idx = dataset_ind.splits['train']
        edge_index, y = dataset_ind.edge_index, dataset_ind.y
        embed_pos, embed_neg, dist_ind = self.forward(dataset_ind)
    
        if self.sampler_type == 'vae':
            x_recon, mu, log_var = self.sampler(embed_pos[train_idx].detach())
            sampler_loss = self.sampler.loss_function(
                x_recon=x_recon, 
                x=embed_pos[train_idx].detach(), 
                mu=mu, 
                log_var=log_var
            )
        elif self.sampler_type == 'diffusion':
            sampler_loss = self.sampler.loss_function(embed_pos[train_idx].detach())

        num_nodes_sample = train_idx.shape[0]
        z_sample = self.sample(
            sample_size=num_nodes_sample,
            device=dataset_ind.x.device)

        dist_sample = self.dist_net(z_sample, self.c).squeeze()
        centroid = self.sample_centroid(embed_pos, self.T)
        self.c = centroid
 
        gen_loss = self.gen_loss(dist_ind[train_idx], dist_sample, reduction)
        logits = self.classifier(embed_pos[train_idx])
        classify_loss = F.cross_entropy(logits, y[train_idx])
        cl_loss = self.encoder.loss_compute(embed_pos, embed_neg, self.c)

        lam_cl = self.config['lam_cl']
        lam_cls = self.config['lam_cls'] 
        lam_gen = self.config['lam_gen']
        lam_sampler = self.config['lam_sampler']
        
        total_loss = (lam_cl * cl_loss + 
                    lam_cls * classify_loss + 
                    lam_gen * gen_loss + 
                    lam_sampler * sampler_loss)
        
        
        return total_loss



    def gen_loss(self, dist_pos, dist_neg, reduction='mean'):

        loss = dist_pos - dist_neg
        loss_reg = dist_pos.pow(2) + dist_neg.pow(2)
        loss = loss + self.lam_r * loss_reg

        if reduction == 'sum':
            return loss.sum()
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'none' or reduction is None:
            return loss
        else:
            raise NotImplementedError

    def sample(self, sample_size, device):
        return self.sampler.sample(sample_size, device)



class SimNet(nn.Module):
    def __init__(self, hidden_channels, shared_proj=None):
        super(SimNet, self).__init__()
        
        self.hid_proj = shared_proj  
        

    def reset_parameters(self):

        pass
    
    def forward(self, x, c):
        x_proj = self.hid_proj(x)  
        cos_sim = torch.nn.functional.cosine_similarity(x_proj, c, dim=-1, eps=1e-8)
        dist = 1 - cos_sim.unsqueeze(-1)
        return dist.squeeze()



class VAESampler(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=64):
        super(VAESampler, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.kl_weight = 1     
        
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var) 
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
    
    def loss_function(self, x_recon, x, mu, log_var):
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') 
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = recon_loss + self.kl_weight * kl_div
        
        return total_loss
    
    @torch.no_grad()
    def sample(self, num_samples, device): 
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples


class DiffusionSampler(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, timesteps=100):
        super(DiffusionSampler, self).__init__()
        
        self.input_dim = input_dim
        self.timesteps = timesteps  
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 16),
        )
        
        self.noise_predictor = nn.Sequential(
            nn.Linear(input_dim + 16, hidden_dim),  
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
       
        self.beta_start = 0.0001  
        self.beta_end = 0.02     
        
        self.beta = torch.linspace(self.beta_start, self.beta_end, timesteps)
        self.alpha = 1. - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward_diffusion(self, x_0, t):
        device = x_0.device
        
        if self.alpha_cumprod.device != device:
            self.beta = self.beta.to(device)
            self.alpha = self.alpha.to(device)
            self.alpha_cumprod = self.alpha_cumprod.to(device)
        
        alpha_cumprod_t = self.alpha_cumprod[t]
        alpha_cumprod_t = alpha_cumprod_t.view(-1, 1)  
        
        noise = torch.randn_like(x_0)
        
        x_t = torch.sqrt(alpha_cumprod_t) * x_0 + torch.sqrt(1 - alpha_cumprod_t) * noise
        
        return x_t, noise
    
    def time_embedding(self, t, device):
        t_float = t.float() / self.timesteps
        t_float = t_float.view(-1, 1).to(device)  
        
        t_emb = self.time_embed(t_float)  
        
        return t_emb
    
    def loss_function(self, x_0, t=None):
        batch_size = x_0.shape[0]
        device = x_0.device
        
        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=device)

        x_t, noise_target = self.forward_diffusion(x_0, t)
        t_emb = self.time_embedding(t, device)  
        x_t_with_t = torch.cat([x_t, t_emb], dim=1)  
        noise_pred = self.noise_predictor(x_t_with_t)
        loss = F.mse_loss(noise_pred, noise_target)
        return loss
    
    @torch.no_grad()
    def sample(self, num_samples, device, delta_scale=0.0):  

        x_t = torch.randn(num_samples, self.input_dim).to(device)
        for t in range(self.timesteps - 1, -1, -1):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            t_emb = self.time_embedding(t_batch, device)
            x_t_with_t = torch.cat([x_t, t_emb], dim=1)
            noise_pred = self.noise_predictor(x_t_with_t)
            
            beta_t = self.beta[t].to(device)
            alpha_t = self.alpha[t].to(device)
            alpha_cumprod_t = self.alpha_cumprod[t].to(device)
            
            if t > 0:
                alpha_cumprod_prev = self.alpha_cumprod[t-1].to(device)
            else:
                alpha_cumprod_prev = torch.tensor(1.0).to(device)
                
            beta_t = beta_t.view(-1, 1)
            alpha_t = alpha_t.view(-1, 1)
            alpha_cumprod_t = alpha_cumprod_t.view(-1, 1)
            alpha_cumprod_prev = alpha_cumprod_prev.view(-1, 1)

            sqrt_recip_alpha_t = 1.0 / torch.sqrt(alpha_t)
            
            if t > 0:
                noise = torch.randn_like(x_t)
                sigma_t = torch.sqrt((1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t) * beta_t)
            else:
                noise = 0
                sigma_t = 0
            
        
            x_t = sqrt_recip_alpha_t * (x_t - beta_t / torch.sqrt(1 - alpha_cumprod_t) * noise_pred) + sigma_t * noise
    
        return x_t

