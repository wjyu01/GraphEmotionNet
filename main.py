'''
Description: 
Author: voicebeer
Date: 2020-09-14 01:01:51
LastEditTime: 2021-12-28 01:46:52
'''
# standard
import torch
import numpy as np
from tqdm import tqdm
import models
import os
import random
from torch.optim import Adam,SGD,RMSprop
from torch.autograd import Variable
from sklearn import preprocessing
import scipy.io as scio
import torch.utils.data as Data
from matplotlib import pyplot as plt
from matplotlib import rcParams
import csv
import pandas as pd


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_GEN_contrast(subject_id, parameter, net_params, source_loaders, target_loader):
    setup_seed(20)
    device = net_params['DEVICE']
    model = models.GEN_contrastNet(net_params).to(device)
    optimizer = RMSprop(model.parameters(), lr=parameter['init_lr'], weight_decay=parameter['weight_decay'])
    best_acc = 0.0
    total_loss_curve = np.zeros((parameter['epochs']))  # 20230816
    total_celoss_curve = np.zeros((parameter['epochs']))
    total_dannloss_curve = np.zeros((parameter['epochs']))
    for epoch in range(parameter['epochs']):
        model.train()
        total_loss, total_num, target_bar = 0.0, 0, tqdm(target_loader)
        source_acc_total, target_acc_total = 0, 0

        total_celoss, total_dannloss = 0.0, 0.0  # 20230816

        train_source_iter = enumerate(source_loaders)
        for data_target, label_target in target_bar:
            _, (data_source, labels_source) = next(train_source_iter)
            data_source, labels_source = data_source.to(device), labels_source.to(device)
            data_target, labels_target = data_target.to(device), label_target.to(device)
            data_source, labels_source = Variable(data_source.cuda()), Variable(labels_source.cuda())
            data_target, labels_target = Variable(data_target.cuda()), Variable(labels_target.cuda())

            pred, domain_loss, Sloss, dloss = model(torch.cat((data_source, data_target)))

            source_pred = pred[0:len(data_source), :]
            target_pred = pred[len(data_source):, :]

            log_prob = torch.nn.functional.log_softmax(source_pred, dim=1)
            celoss = -torch.sum(log_prob * labels_source) / len(labels_source)
            loss = celoss + domain_loss + Sloss + dloss
            source_scores = source_pred.detach().argmax(dim=1)
            source_acc = (source_scores == labels_source.argmax(dim=1)).float().sum().item()
            source_acc_total += source_acc
            target_scores = target_pred.detach().argmax(dim=1)
            target_acc = (target_scores == labels_target.argmax(dim=1)).float().sum().item()
            target_acc_total += target_acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_num += parameter['batch_size']
            total_loss += loss.item() * parameter['batch_size']
            epoch_train_loss = total_loss / total_num
            # 20230816
            total_celoss += celoss.item() * parameter['batch_size']
            epoch_train_celoss = total_celoss / total_num
            total_dannloss += domain_loss.item() * parameter['batch_size']
            epoch_train_dannloss = total_dannloss / total_num

            target_bar.set_description('sub:{} Train Epoch: [{}/{}] Loss: {:.4f} source_acc:{:.2f}% target_acc:{:.2f}%'
                                       .format(subject_id, epoch + 1, parameter['epochs'], epoch_train_loss,
                                               source_acc_total / total_num * 100,
                                               target_acc_total / total_num * 100))
        total_loss_curve[epoch] = epoch_train_loss
        total_celoss_curve[epoch] = epoch_train_celoss
        total_dannloss_curve[epoch] = epoch_train_dannloss

        if best_acc < (target_acc_total / total_num):
            best_acc = (target_acc_total / total_num)
        # scheduler.step(epoch_train_loss)
        # os.chdir('E:\\model_result')
        # torch.save(model.state_dict(),'model'+str(subject_id)+'.pkl')
    return best_acc, total_loss_curve, total_celoss_curve, total_dannloss_curve


def test_GEN(subject_id, epoch, model, target_loader, parameter):
    model.eval()
    target_acc_total, total_num, target_bar = 0.0, 0, tqdm(target_loader)
    for data_target, label_target in target_bar:
        pred, _, _, _ = model(data_target)
        target_scores = pred.detach().argmax(dim=1)
        target_acc = (target_scores == label_target.argmax(dim=1)).float().sum().item()
        target_acc_total += target_acc
        total_num += parameter['batch_size']
        target_bar.set_description('sub:{} Train Epoch: [{}/{}] target_acc:{:.2f}%'
                                       .format(subject_id, epoch+1, parameter['epochs'],
                                               target_acc_total/total_num * 100))
    return target_acc_total / total_num

def AddContext(x, context, label=False, dtype='float32'):
    ret = []
    assert context % 2 == 1, "context value error."

    cut = int(context / 2)
    if label:
        for p in range(len(x)):
            tData = x[p][cut:x[p].shape[0] - cut]
            ret.append(tData)
            # print(tData.shape)
    else:
        for p in range(len(x)):
            tData = np.zeros([x[p].shape[0] - 2 * cut, context, x[p].shape[1], x[p].shape[2]], dtype=dtype)
            for i in range(cut, x[p].shape[0] - cut):
                tData[i - cut] = x[p][i - cut:i + cut + 1]

            # print(tData.shape)
            ret.append(tData)
    return ret

def get_dataset(test_id, session):
    session =session+1
    path = '/root/autodl-tmp/dataset/feature_for_net_session' + str(session) + '_LDS_de'
    os.chdir(path)
    feature_list_source_labeled = []
    label_list_source_labeled = []
    feature_list_target = []
    label_list_target = []
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    #video_time = [235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206]
    index = 0
    for info in os.listdir(path):
        domain = os.path.abspath(path)
        info_ = os.path.join(domain, info)  # 将路径与文件名结合起来就是每个文件的完整路径
        if session == 1:
            feature = scio.loadmat(info_)['dataset_session1']['feature'][0, 0]
            label = scio.loadmat(info_)['dataset_session1']['label'][0, 0]
        elif session == 2:
            feature = scio.loadmat(info_)['dataset_session2']['feature'][0, 0]
            label = scio.loadmat(info_)['dataset_session2']['label'][0, 0]
        else:
            feature = scio.loadmat(info_)['dataset_session3']['feature'][0, 0]
            label = scio.loadmat(info_)['dataset_session3']['label'][0, 0]

        feature = min_max_scaler.fit_transform(feature).astype('float32')
        feature = feature.reshape(feature.shape[0], 62, 5, order='F')
        trial_list = []
        trial_label_list = []
        '''
        for video in range(len(video_time)):
            if video==0:
                trial = feature[0:np.cumsum(video_time[0:video + 1])[-1], :]
                trial_label = label[0:np.cumsum(video_time[0:video + 1])[-1], :]
            else:
                trial = feature[np.cumsum(video_time[0:video])[-1]:np.cumsum(video_time[0:video+1])[-1],:]
                trial_label = label[np.cumsum(video_time[0:video])[-1]:np.cumsum(video_time[0:video + 1])[-1], :]
            trial_list.append(trial)
            trial_label_list.append(trial_label)
        '''
        feature = AddContext(trial_list,3)
        label = AddContext(trial_label_list, 3, label=True)
        feature = np.vstack(feature)
        label = np.vstack(label)

        one_hot_label_mat = np.zeros((len(label), 3))
        for i in range(len(label)):
            if label[i] == 0:
                one_hot_label = [1, 0, 0]
                one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                one_hot_label_mat[i, :] = one_hot_label
            if label[i] == 1:
                one_hot_label = [0, 1, 0]
                one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                one_hot_label_mat[i, :] = one_hot_label
            if label[i] == 2:
                one_hot_label = [0, 0, 1]
                one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
                one_hot_label_mat[i, :] = one_hot_label

        if index != test_id:
            ## source labeled data
            feature_labeled = feature
            label_labeled = one_hot_label_mat
            feature_list_source_labeled.append(feature_labeled)
            label_list_source_labeled.append(label_labeled)

        else:
            ## target labeled data
            feature_list_target.append(feature)
            label_list_target.append(one_hot_label_mat)
            label = one_hot_label_mat
        index += 1

    source_feature_labeled, source_label_labeled = np.vstack(feature_list_source_labeled), np.vstack(label_list_source_labeled)

    target_feature = feature_list_target[0]
    target_label = label_list_target[0]

    target_set = {'feature': target_feature, 'label': target_label}
    source_set_labeled = {'feature': source_feature_labeled, 'label': source_label_labeled}

    return target_set, source_set_labeled




def cross_subject(target_set, source_set_labeled, session_id, subject_id, parameter, net_params):
    setup_seed(20)
    torch_dataset_test = Data.TensorDataset(torch.from_numpy(target_set['feature']), torch.from_numpy(target_set['label']))
    torch_dataset_source_labeled = Data.TensorDataset(torch.from_numpy(source_set_labeled['feature']), torch.from_numpy(source_set_labeled['label']))
    source_loaders = torch.utils.data.DataLoader(dataset=torch_dataset_source_labeled,
                                                 batch_size=parameter['batch_size'],
                                                 shuffle=True,
                                                 drop_last=True)
    target_loader = torch.utils.data.DataLoader(dataset=torch_dataset_test,
                                                batch_size=parameter['batch_size'],
                                                shuffle=True,
                                                drop_last=True)

   
    acc = train_GEN_contrast(subject_id,parameter,net_params,source_loaders=source_loaders,target_loader=target_loader)
    return acc


def main(parameter,net_params):
   # data preparation
    if not os.path.exists('figures'):
        os.mkdir('figures')
    if not os.path.exists('csvfile'):
        os.mkdir('csvfile')


    setup_seed(20)
    print('Model name: MS-MDAER. Dataset name: ', parameter['dataset_name'])
    print('BS: {}, epoch: {}'.format(parameter['batch_size'], parameter['epochs']))
    # store the results


    # for session_id_main in range(3):
    session_id = 0
    for subject_id in range(15):
        csub = []
        loss_curve = []  # 20230816
        celoss_curve = []
        dannloss_curve = []

        target_set, source_set_labeled = get_dataset(subject_id, session_id)
        result = cross_subject(target_set, source_set_labeled, session_id, subject_id, parameter, net_params)
        csub.append(result[0])
        loss_curve.append(result[1])# 20230816
        celoss_curve.append(result[2])
        dannloss_curve.append(result[3])
        loss_curve = [num for sublist in loss_curve for num in sublist]
        celoss_curve = [num for sublist in celoss_curve for num in sublist]
        dannloss_curve = [num for sublist in dannloss_curve for num in sublist]

        loss_pd = pd.DataFrame(loss_curve)
        loss_pd.to_csv(os.path.join('/root','csvfile', f'id{subject_id + 1}_loss'))

        celoss_pd = pd.DataFrame(celoss_curve)
        celoss_pd.to_csv(os.path.join('/root', 'csvfile', f'id{subject_id + 1}_celoss'))

        dannloss_pd = pd.DataFrame(dannloss_curve)
        dannloss_pd.to_csv(os.path.join('/root', 'csvfile', f'id{subject_id + 1}_dannloss'))

        #print(loss_curve)
        plt.rc('font',family='Times New Roman')
        #xs = range(parameter['epochs'])
        plt.figure()
        plt.plot(loss_curve)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        #plt.legend()
        plt.savefig(os.path.join('/root','figures', f'id{subject_id + 1}_loss.png'))

        plt.figure()
        plt.plot(celoss_curve)
        plt.xlabel('Epoch')
        plt.ylabel('CeLoss')
        #plt.legend()
        plt.savefig(os.path.join('/root','figures', f'id{subject_id + 1}_CeLoss.png'))


        plt.figure()
        plt.plot(dannloss_curve)
        plt.xlabel('Epoch')
        plt.ylabel('DannLoss')
        #plt.legend()
        plt.savefig(os.path.join('/root','figures', f'id{subject_id + 1}_DannLoss.png'))

    print("Cross-subject: ", csub)
    return csub, loss_curve, celoss_curve, dannloss_curve


parameter = {'dataset_name':'seed3','epochs':1000, 'batch_size':96, 'init_lr':1e-3, 'weight_decay':1e-2}
net_params = {'GLalpha': 1e-2, 'node_feature_hidden1': 5, 'node_feature_hidden2': 5, 'adv_alpha': 1, 'aug_type': 'nn',
            'in_dim': 5, 'hidden_dim': 5, 'out_dim': 5, 'in_feat_dropout': 0.0, 'dropout': 0, 'n_layers': 2,
            'readout': 'mean', 'graph_norm': True, 'batch_norm': True, 'residual': True, 'category_number': 3,
             'DEVICE': 'cuda:0', 'K':2, 'num_of_timesteps': 3, 'num_of_vertices': 62, 'num_of_features': 5,
             }
csub,loss_curve,celoss_curve,dannloss_curve = main(parameter, net_params)









