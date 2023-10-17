from random import randint

import numpy as np
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import torch
import torch.nn as nn
from torch.utils.data import dataset

from models import EMModule, RecurrentContextModule
import utils


""" Data generation functions """


class CategoryDataset(dataset.Dataset):
    def __init__(self, n_samples_per_context, contexts_to_load, noise=0):
        super().__init__()
        xs = torch.tensor(np.load('data/exp3_reps.npy')).float()
        ys = torch.tensor(np.load('data/exp3_features.npy')).float()
        contexts = torch.tensor(np.load('data/exp3_context.npy')).float()
        xs = torch.concatenate([xs]*20)
        ys = torch.concatenate([ys]*20)
        contexts = torch.concatenate([contexts]*20)
        xs += torch.randn_like(xs)*noise

        xs_c1, ys_c1 = xs[contexts==1], ys[contexts==1]
        xs_c2, ys_c2 = xs[contexts==2], ys[contexts==2]
        xs = torch.dstack([xs_c1, xs_c2])
        ys = torch.dstack([ys_c1, ys_c2])
        n_samples_per_context = n_samples_per_context
        item_indices = np.random.choice(xs.shape[0], sum(n_samples_per_context), replace=True)
        task_names = [0,1]
        task_indices = [task_names.index(name) for name in contexts_to_load]
        context_indices = np.repeat(np.array(task_indices), n_samples_per_context)
        self.xs = xs[item_indices, :, context_indices]
        ys = ys[item_indices, context_indices, context_indices].numpy()
        self.ys = utils.thermometer_encode(ys, 5)
        self.contexts = utils.one_hot_encode(context_indices, len(task_names))


    def __len__(self):
        return len(self.xs)


    def __getitem__(self, idx):
        return self.xs[idx], self.contexts[idx], self.ys[idx]


def gen_data_loader(paradigm, noise=0):
    if paradigm=='Blocked':
        contexts_to_load = [0,1] + [randint(0,1) for _ in range(201)]
        n_samples_per_context = [200,200] + [1]*201
    elif paradigm == 'Interleaved':
        contexts_to_load = [randint(0,1) for _ in range(601)]
        n_samples_per_context = [1]*601
    ds = CategoryDataset(n_samples_per_context, contexts_to_load, noise=noise)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)


""" Model preparation functions """


def prep_semantic_network(ffnet, semantic_d, context_d):
    with torch.no_grad():
        new_weights = torch.zeros((semantic_d,context_d))
        for i in range(0,semantic_d,semantic_d//context_d):
            new_weights[i:i+semantic_d//context_d,i//(semantic_d//context_d)] = -3
        ffnet[0].weight.copy_(torch.cat([ffnet[0].weight[:,:-context_d],new_weights],dim=-1))
    return ffnet


def prep_recurrent_network(rnet, state_d, context_d, persistance=-.6):
    with torch.no_grad():
        rnet.state_to_hidden.weight.copy_(torch.eye(state_d,dtype=torch.float))
        rnet.state_to_hidden.bias.zero_()
        rnet.hidden_to_hidden.weight.zero_()
        rnet.hidden_to_hidden.bias.zero_()
        rnet.state_to_hidden_wt.weight.zero_()
        rnet.state_to_hidden_wt.bias.copy_(torch.ones((len(rnet.state_to_hidden_wt.bias),),dtype=torch.float)*persistance)
        rnet.hidden_to_hidden_wt.weight.zero_()
        rnet.hidden_to_hidden_wt.bias.zero_()
        # Set hidden to context weights as an identity matrix if possible
        if state_d == context_d:
            rnet.hidden_to_context.weight.copy_(torch.eye(state_d,dtype=torch.float))
        rnet.hidden_to_context.bias.zero_()
    # Freeze recurrent weights to stabilize training
    for p in rnet.parameters():
        p.requires_grad = False
    rnet.hidden_to_context.weight.requires_grad = True
    return rnet


def gen_model(params):
    semantic_pathway = nn.Sequential(
                nn.Linear(params.state_d+params.context_d,params.semantic_d),
                nn.Sigmoid(),
                nn.Linear(params.semantic_d,params.output_d),
                nn.Sigmoid()
            )
    context_module = RecurrentContextModule(params.state_d, params.state_d, params.context_d)
    em_module = EMModule(params.temperature)

    semantic_pathway = prep_semantic_network(semantic_pathway, params.semantic_d, params.context_d)
    context_module = prep_recurrent_network(context_module, params.state_d, params.context_d, params.persistance)
    return semantic_pathway, context_module, em_module


""" Task performance functions """


def calc_accuracy(pred, true):
    return ((pred.sum()>2.5)==(true.sum()>2.5)).float().item()


def run_experiment(params):
    performance_data = {'seed':[],'paradigm':[],'trial':[],'accuracy':[],'pathway':[]}
    model_data = []
    loss_fn = nn.MSELoss()
    for seed in range(params.n_participants):
        utils.set_random_seed(seed)
        for training_paradigm in ['Interleaved','Blocked']:
            data_loader = gen_data_loader(training_paradigm)
            semantic_pathway, context_module, em_module = gen_model(params)
            optimizer = torch.optim.SGD([{'params':semantic_pathway.parameters(),'lr':params.semantic_lr},
                                        {'params':context_module.parameters(),'lr':params.episodic_lr}])

            sm_accuracy, em_accuracy = [], []
            for trial, (x,_,y) in enumerate(data_loader):
                context = context_module(x)
                pred_sm = semantic_pathway(torch.cat([x,context],dim=-1))
                loss = loss_fn(pred_sm,y)
                if trial > 0:
                    pred_em = em_module(x,context)
                    loss += loss_fn(pred_em,y)
                else:
                    pred_em = torch.zeros([1,params.output_d]).float()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    sm_accuracy.append(calc_accuracy(pred_sm,y))
                    em_accuracy.append(calc_accuracy(pred_em,y))
                    em_module.write(x,context,y)

            sm_accuracy = np.concatenate([[0]*49,np.convolve(sm_accuracy, np.ones(50)/50, mode='valid')])
            em_accuracy = np.concatenate([[0]*49,np.convolve(em_accuracy, np.ones(50)/50, mode='valid')])
            combined_accuracy = (sm_accuracy+em_accuracy)/2
            performance_data['accuracy'].extend(np.concatenate([sm_accuracy,em_accuracy,combined_accuracy]))
            performance_data['pathway'].extend(['Semantic']*len(sm_accuracy)+['Episodic']*len(em_accuracy)+['Combined']*len(combined_accuracy))
            performance_data['seed'].extend([seed]*len(combined_accuracy)*3)
            performance_data['paradigm'].extend([training_paradigm]*len(combined_accuracy)*3)
            performance_data['trial'].extend(list(range(len(combined_accuracy)))*3)
            model_data.append({'paradigm':training_paradigm,'seed':seed,'semantic_pathway':semantic_pathway,'context_module':context_module,'em_module':em_module})
    return performance_data, model_data


""" Analysis functions """


def get_template_rdm(rdm_name):
    all_features = np.load('data/exp3_features.npy')
    context = np.load('data/exp3_context.npy')-1
    if rdm_name == 'shared':
        reps = np.concatenate([all_features,context[:,np.newaxis]],axis=-1)
        sim = -euclidean_distances(reps)[np.triu_indices(len(reps),k=1)]
    elif rdm_name == 'separate':
        reps = np.zeros((len(all_features),2))
        reps[context==0,0] = all_features[context==0,0]
        reps[context==1,1] = all_features[context==1,1]
        reps_cosine = np.concatenate([utils.thermometer_encode(reps[:,0],5),utils.thermometer_encode(reps[:,1],5)],axis=-1)
        sim = cosine_similarity(reps_cosine)[np.triu_indices(len(reps_cosine),k=1)]
    else:
        raise ValueError(f'Invalid rdm_name {rdm_name}. Try one of ["shared", "separate"].')
    return sim


def get_embedding(model_data, x):
    contexts = model_data['context_module'].get_terminal_state(x).detach()
    in_data = torch.cat([x,contexts],dim=-1)
    embedding = model_data['semantic_pathway'][:2](in_data).detach().numpy()
    return embedding


def get_rotated_mds(embedding, rotation_type='angle'):
    mds_reps = MDS(3,random_state=0).fit_transform(embedding)
    if rotation_type == 'angle':
        A = mds_reps[25]
        B = mds_reps[29]
        C = mds_reps[-5]
        AB = B-A
        AC = C-A
        normal_vector = np.cross(AB,AC)
        normal_vector /= np.linalg.norm(normal_vector)
        nx, ny, nz = normal_vector
        angle = np.arccos(nx / np.linalg.norm([nx,ny,nz]))
        axis = np.cross([1, 0, 0], [nx, ny, nz])
        axis /= np.linalg.norm(axis)
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        rotation_matrix = np.array([
            [cos_theta + axis[0]**2 * (1 - cos_theta), axis[0] * axis[1] * (1 - cos_theta) - axis[2] * sin_theta, axis[0] * axis[2] * (1 - cos_theta) + axis[1] * sin_theta],
            [axis[1] * axis[0] * (1 - cos_theta) + axis[2] * sin_theta, cos_theta + axis[1]**2 * (1 - cos_theta), axis[1] * axis[2] * (1 - cos_theta) - axis[0] * sin_theta],
            [axis[2] * axis[0] * (1 - cos_theta) - axis[1] * sin_theta, axis[2] * axis[1] * (1 - cos_theta) + axis[0] * sin_theta, cos_theta + axis[2]**2 * (1 - cos_theta)]
        ])
        rot_reps = np.einsum('ij,kj->ki',rotation_matrix,mds_reps)
    elif rotation_type == 'flip':
        flipped_reps = mds_reps.copy()
        flipped_reps[:,:2] *= -1
        rot_reps = np.zeros_like(flipped_reps)
        rot_reps[:,0] = flipped_reps[:,2]
        rot_reps[:,1] = flipped_reps[:,0]
        rot_reps[:,2] = flipped_reps[:,1]
    else:
        raise ValueError(f'Invalid rotation_type {rotation_type}. Try one of ["angle", "flip"].')
    return rot_reps