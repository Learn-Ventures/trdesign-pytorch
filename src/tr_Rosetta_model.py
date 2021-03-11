from utils import *
import torch
import torch.nn.functional as F
from torch import nn
from pathlib import Path

'''

Credits to @lucidrains for this implementation
https://github.com/lucidrains

'''

# 1-hot MSA to PSSM
def msa2pssm(msa1hot, w):
    beff = w.sum()
    f_i = (w[:, None, None] * msa1hot).sum(dim=0) / beff + 1e-9
    h_i = (-f_i * torch.log(f_i)).sum(dim=1)
    return torch.cat((f_i, h_i[:, None]), dim=1)

# reweight MSA based on cutoff
def reweight(msa1hot, cutoff):
    id_min = msa1hot.shape[1] * cutoff
    id_mtx = torch.einsum('ikl,jkl->ij', msa1hot, msa1hot)
    id_mask = id_mtx > id_min
    w = 1. / id_mask.float().sum(dim=-1)
    return w

# shrunk covariance inversion
def fast_dca(msa1hot, weights, penalty = 4.5):
    device = msa1hot.device
    nr, nc, ns = msa1hot.shape
    x = msa1hot.view(nr, -1)
    num_points = weights.sum() - torch.sqrt(weights.mean())

    mean = (x * weights[:, None]).sum(dim=0, keepdims=True) / num_points
    x = (x - mean) * torch.sqrt(weights[:, None])

    cov = (x.t() @ x) / num_points
    cov_reg = cov + torch.eye(nc * ns).to(d()) * penalty / torch.sqrt(weights.sum())

    inv_cov = torch.inverse(cov_reg)
    x1 = inv_cov.view(nc, ns, nc, ns)
    x2 = x1.transpose(1, 2).contiguous()
    features = x2.reshape(nc, nc, ns * ns)

    x3 = torch.sqrt((x1[:, :-1, :, :-1] ** 2).sum(dim=(1, 3))) * (1 - torch.eye(nc).to(d()))
    apc = x3.sum(dim=0, keepdims=True) * x3.sum(dim=1, keepdims=True) / x3.sum()
    contacts = (x3 - apc) * (1 - torch.eye(nc).to(d()))
    return torch.cat((features, contacts[:, :, None]), dim=2)

# read A3M and convert letters into
# integers in the 0..20 range
def parse_a3m(filename):
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    seqs = [line.strip().translate(table) for line in open(filename, 'r') if line[0] != '>']
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)

    # convert letters into numbers
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20
    return msa

def prep_seq(a3m, wmin=0.8, ns=21):
    nrow, ncol = a3m.shape
    msa1hot = F.one_hot(a3m, ns).float().to(d())
    w = reweight(msa1hot, wmin).float().to(d())

    # 1d sequence
    f1d_seq = msa1hot[0, :, :20].float()
    f1d_pssm = msa2pssm(msa1hot, w)

    f1d = torch.cat((f1d_seq, f1d_pssm), dim=1)
    f1d = f1d[None, :, :].reshape((1, ncol, 42))

    # 2d sequence
    f2d_dca = fast_dca(msa1hot, w) if nrow > 1 else torch.zeros((ncol, ncol, 442)).float()
    f2d_dca = f2d_dca[None, :, :, :]
    f2d_dca = f2d_dca.to(d())

    f2d = torch.cat((
        f1d[:, :, None, :].repeat(1, 1, ncol, 1), 
        f1d[:, None, :, :].repeat(1, ncol, 1, 1),
        f2d_dca
    ), dim=-1)

    f2d = f2d.view(1, ncol, ncol, 442 + 2*42)
    return f2d.permute((0, 3, 2, 1)), msa1hot

def preprocess(msa_file = None, wmin=0.8, ns=21, use_random_seq = False):
    if use_random_seq:
        a3m = torch.randint(0, 20, (1, 67), device = d(), requires_grad = False)
    elif msa_file is not None:
        a3m = torch.from_numpy(parse_a3m(msa_file)).long()

    return prep_seq(a3m, wmin=wmin, ns=ns)

# model code

def instance_norm(filters, eps=1e-6, **kwargs):
    return nn.InstanceNorm2d(filters, affine=True, eps=eps, **kwargs)

def conv2d(in_chan, out_chan, kernel_size, dilation=1, **kwargs):
    padding = dilation * (kernel_size - 1) // 2
    return nn.Conv2d(in_chan, out_chan, kernel_size, padding=padding, dilation=dilation, **kwargs)

def elu():
    return nn.ELU(inplace=True)

class trRosettaNetwork(nn.Module):
    def __init__(self, filters=64, kernel=3, num_layers=61):
        super().__init__()
        self.filters = filters
        self.kernel = kernel
        self.num_layers = num_layers

        self.first_block = nn.Sequential(
            conv2d(442 + 2 * 42, filters, 1),
            instance_norm(filters),
            elu()
        )

        # stack of residual blocks with dilations
        cycle_dilations = [1, 2, 4, 8, 16]
        dilations = [cycle_dilations[i % len(cycle_dilations)] for i in range(num_layers)]

        self.layers = nn.ModuleList([nn.Sequential(
            conv2d(filters, filters, kernel, dilation=dilation),
            instance_norm(filters),
            elu(),
            nn.Dropout(p=0.15),
            conv2d(filters, filters, kernel, dilation=dilation),
            instance_norm(filters)
        ) for dilation in dilations])

        self.activate = elu()

        # conv to anglegrams and distograms
        self.to_prob_theta = nn.Sequential(conv2d(filters, 25, 1), nn.Softmax(dim=1))
        self.to_prob_phi   = nn.Sequential(conv2d(filters, 13, 1), nn.Softmax(dim=1))
        self.to_distance   = nn.Sequential(conv2d(filters, 37, 1), nn.Softmax(dim=1))
        self.to_prob_bb    = nn.Sequential(conv2d(filters, 3,  1), nn.Softmax(dim=1))
        self.to_prob_omega = nn.Sequential(conv2d(filters, 25, 1), nn.Softmax(dim=1))
 
    def forward(self, x):
        x = self.first_block(x)

        for layer in self.layers:
            x = self.activate(x + layer(x))
            
        prob_theta = self.to_prob_theta(x)      # anglegrams for theta
        prob_phi = self.to_prob_phi(x)          # anglegrams for phi

        x = 0.5 * (x + x.permute((0,1,3,2)))    # symmetrize

        prob_distance = self.to_distance(x)     # distograms
        #prob_bb = self.to_prob_bb(x)            # beta-strand pairings (not used)
        prob_omega = self.to_prob_omega(x)      # anglegrams for omega

        return prob_theta, prob_phi, prob_distance, prob_omega

class trRosettaEnsemble(nn.Module):
    def __init__(self, trRosetta_model_dir, use_n_models = np.inf):
        super().__init__()

        self.model_dir   = trRosetta_model_dir
        self.model_paths = [*Path(self.model_dir).glob('*.pt')]

        if len(self.model_paths) == 0:
            raise 'No model files can be found'

        if use_n_models < len(self.model_paths):
            self.model_paths = self.model_paths[:use_n_models]

        self.n_models = len(self.model_paths)
        self.load()

    def load(self):
        self.models       = [0] * len(self.model_paths)
        #self.cuda_streams = [torch.cuda.Stream() for i in self.model_paths]
        
        for i, model_path in enumerate(self.model_paths):
            print("Loading %s..." %model_path)
            self.models[i] = trRosettaNetwork() #.share_memory()
            self.models[i].load_state_dict(torch.load(model_path, map_location=torch.device(d())))
            self.models[i].to(d()).eval()
    
    def forward(self, x, use_n_models = None, dump_distograms_path = None):
        if use_n_models is None:
            use_n_models = self.n_models

        # TODO: Make this forward pass use parallel GPU threads
        outputs = []
        for i, structure_model in enumerate(self.models[:use_n_models]):
            #with torch.cuda.stream(self.cuda_streams[i]):
            outputs.append(structure_model(x))

        if dump_distograms_path is not None:
            for i, output in enumerate(outputs):
                pt, pp, pd, po = output
                distogram_distribution = pd.cpu().detach().numpy()
                distogram = distogram_distribution_to_distogram(distogram_distribution)
                plot_distogram(distogram, os.path.join(dump_distograms_path, 'dist_model_%02d.jpg' %i))

        averaged_outputs = [torch.stack(model_output).mean(axis=0) for model_output in zip(*outputs)]
        return averaged_outputs

