import os
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, confusion_matrix

from torchvision import transforms
from torch.utils.data import Dataset
# import pickle5 as pickle
import pickle 


import torch.nn as nn

device=  torch.device("cuda" if torch.cuda.is_available() else "cpu")

from fairlearn.metrics import MetricFrame, demographic_parity_difference

tfms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tensor_to_numpy_func = lambda   x: x.cpu().numpy() if torch.is_tensor(x) else x 

class CelebA(Dataset):
    def __init__(self, dataframe, folder_dir, target_id, transform=None, gender=None, target=None):
        self.dataframe = dataframe
        self.folder_dir = folder_dir
        self.target_id = target_id
        self.transform = transform
        self.file_names = dataframe.index
        self.labels = np.concatenate(dataframe.labels.values).astype(float)
        gender_id = 20

        if gender is not None:
            if target is not None:
                label_np = np.concatenate(dataframe.labels.values)
                gender_idx = np.where(label_np[:, gender_id] == gender)[0]
                target_idx = np.where(label_np[:, target_id] == target)[0]
                idx = list(set(gender_idx) & set(target_idx))
                self.file_names = self.file_names[idx]
                self.labels = np.concatenate(dataframe.labels.values[idx]).astype(float)
            else:
                label_np = np.concatenate(dataframe.labels.values)
                gender_idx = np.where(label_np[:, gender_id] == gender)[0]
                self.file_names = self.file_names[gender_idx]
                self.labels = np.concatenate(dataframe.labels.values[gender_idx]).astype(float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.folder_dir, self.file_names[index]))
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label[self.target_id]
    
class CelebA_gender(Dataset):
    def __init__(self, dataframe, folder_dir, target_id, transform=None, gender=None, target=None):
        self.dataframe = dataframe
        self.folder_dir = folder_dir
        self.target_id = target_id
        self.transform = transform
        self.file_names = dataframe.index
        self.labels = np.concatenate(dataframe.labels.values).astype(float)
        gender_id = 20

        if gender is not None:
            if target is not None:
                label_np = np.concatenate(dataframe.labels.values)
                gender_idx = np.where(label_np[:, gender_id] == gender)[0]
                target_idx = np.where(label_np[:, target_id] == target)[0]
                idx = list(set(gender_idx) & set(target_idx))
                self.file_names = self.file_names[idx]
                self.labels = np.concatenate(dataframe.labels.values[idx]).astype(float)
            else:
                label_np = np.concatenate(dataframe.labels.values)
                gender_idx = np.where(label_np[:, gender_id] == gender)[0]
                self.file_names = self.file_names[gender_idx]
                self.labels = np.concatenate(dataframe.labels.values[gender_idx]).astype(float)
                
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.folder_dir, self.file_names[index]))
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label[self.target_id], label[20]


class CelebA_latent_codes(Dataset):
    def __init__(self, target_id=2, mode="train"):
        if target_id == 2:
            if mode == "train":
                self.ori_path = f"latent_codes/train_latent_codes.pkl"
            elif mode == "val":
                self.ori_path = f"latent_codes/val_latent_codes.pkl"
            elif mode == "test":
                self.ori_path = f"latent_codes/test_latent_codes.pkl"
        with open(self.ori_path, 'rb') as handle:
            ori_data = pickle.load(handle)
        strategy = "perp_on_hyperplane"
        self.targets = np.concatenate([ori_data["targets"], ori_data["targets"]], axis=0)
        self.gender_labels = np.concatenate([ori_data["gender_labels"], 1-ori_data["gender_labels"]], axis=0)
        self.feats = np.concatenate([ori_data ["feats"], ori_data[strategy]], axis=0)
        print(len(ori_data["feats"]), len(ori_data[strategy]))
        print(len(self.targets), len(self.gender_labels), len(self.feats))
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.feats[index].astype(np.float32), self.targets[index], self.gender_labels[index]


class CelebA_latent_codes_seperate(Dataset):
        def __init__(self, target_id=2):
            if target_id == 2:
                self.ori_path = f"latent_codes/train_latent_codes.pkl"
            with open(self.ori_path, 'rb') as handle:
                ori_data = pickle.load(handle)
            strategy = "perp"
            self.targets = ori_data["targets"]
            self.gender_labels = ori_data["gender_labels"]
            self.feats = ori_data["feats"]
            self.targets2 = ori_data["targets"]
            self.gender_labels2 = 1-ori_data["gender_labels"]
            self.feats2 = ori_data[strategy]

            # print(len(ori_data["feats"]), len(ori_data[strategy]))
            # print(len(self.targets), len(self.gender_labels), len(self.feats))

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return {"data1": self.feats[index].astype(np.float32), "data1_target": self.targets[index], "data1_gender_label": self.gender_labels[index],
                    "data2": self.feats2[index].astype(np.float32), "data2_target": self.targets2[index], "data2_gender_label": self.gender_labels2[index]}


def get_loader(df, data_path, target_id, batch_size, gender=None, target=None):
    dl = CelebA(df, data_path, target_id, transform=tfms, gender=gender, target=target)

    if 'train' in data_path:
        dloader = torch.utils.data.DataLoader(dl, shuffle=True, batch_size=batch_size, num_workers=3, drop_last=True)
    else:
        dloader = torch.utils.data.DataLoader(dl, shuffle=False, batch_size=batch_size, num_workers=3)

    return dloader

def get_loader_gender(df, data_path, target_id, batch_size, gender=None, target=None):
    dl = CelebA_gender(df, data_path, target_id, transform=tfms, gender=gender, target=target)

    if 'train' in data_path:
        dloader = torch.utils.data.DataLoader(dl, shuffle=True, batch_size=batch_size, num_workers=3, drop_last=True)
    else:
        dloader = torch.utils.data.DataLoader(dl, shuffle=False, batch_size=batch_size, num_workers=3)

    return dloader

def get_loader_gender_unshuffle(df, data_path, target_id, batch_size, gender=None, target=None):
    dl = CelebA_gender(df, data_path, target_id, transform=tfms, gender=gender, target=target)

    if 'train' in data_path:
        dloader = torch.utils.data.DataLoader(dl, shuffle=False, batch_size=batch_size, num_workers=3)
    else:
        dloader = torch.utils.data.DataLoader(dl, shuffle=False, batch_size=batch_size, num_workers=3)

    return dloader

def get_loader_latent_codes_unshuffle(target_id, batch_size, mode):
    dl = CelebA_latent_codes(target_id, mode=mode)
    dloader = torch.utils.data.DataLoader(dl, shuffle=False, batch_size=batch_size, num_workers=3, drop_last=False)
    return dloader

def get_loader_latent_codes_seperate(df, data_path, target_id, batch_size, gender=None, target=None):
    dl = CelebA_latent_codes_seperate(target_id)
    dloader = torch.utils.data.DataLoader(dl, shuffle=True, batch_size=batch_size, num_workers=3, drop_last=True)
    return dloader

def evaluate(model, model_linear, dataloader):
    y_scores = []
    y_true = []
    for i, (inputs, target) in enumerate(dataloader):
        inputs, target = inputs.to(device), target.float().to(device)

        feat = model(inputs)
        pred = model_linear(feat).detach()

        y_scores.append(pred[:, 0].cpu().numpy())
        y_true.append(target.cpu().numpy())

    y_scores = np.concatenate(y_scores)
    y_true = np.concatenate(y_true)
    ap = average_precision_score(y_true, y_scores)
    return ap, np.mean(y_scores)

def evaluate_mask(model, model_linear, dataloader, mask):
    y_scores = []
    y_scores_masked = []
    y_true = []
    for i, (inputs, target) in enumerate(dataloader):
        inputs, target = inputs.to(device), target.float().to(device)

        feat = model(inputs)
        # print("feat:", feat.shape)
        mask_layer = torch.ones_like(feat)
        mask_layer[:, mask, :, :] = 0
        feat_masked = feat * mask_layer
        pred = model_linear(feat).detach()
        pred_masked = model_linear(feat_masked).detach()

        y_scores.append(pred[:, 0].cpu().numpy())
        y_scores_masked.append(pred_masked[:, 0].cpu().numpy())
        y_true.append(target.cpu().numpy())

    y_scores = np.concatenate(y_scores)
    y_scores_masked = np.concatenate(y_scores_masked)
    diff = np.linalg.norm((y_scores_masked - y_scores), ord=2)
    y_true = np.concatenate(y_true)
    ap = average_precision_score(y_true, y_scores)
    ap_masked = average_precision_score(y_true, y_scores_masked)
    return ap, ap_masked, diff, np.mean(y_scores)

def generate_embedding(model, avg, dataloader):
    feats = []
    targets = []
    gender_labels = []
    model.eval()
    for i, (inputs, target, gender) in enumerate(dataloader):
        print("Batch Index:", i)
        inputs, target, gender = inputs.to(device), target.float().to(device), gender.float().to(device)
        feat = model(inputs)

        feat = avg(feat).view(-1, 512)
        feats.append(feat.cpu().numpy())
        targets.append(target.cpu().numpy())
        gender_labels.append(gender.cpu().numpy())

    feats = np.concatenate(feats)
    targets = np.concatenate(targets)
    gender_labels = np.concatenate(gender_labels)

    return feats, targets, gender_labels


def generate_embedding_last_layer(model, linear_layer, dataloader):
    feats = []
    targets = []
    gender_labels = []
    model.eval()
    for i, (inputs, target, gender) in enumerate(dataloader):
        print("Batch Index:", i)
        inputs, target, gender = inputs.to(device), target.float().to(device), gender.float().to(device)
        feat = model(inputs)
        feat = linear_layer(feat)
        feats.append(feat.cpu().numpy())
        targets.append(target.cpu().numpy())
        gender_labels.append(gender.cpu().numpy())

    feats = np.concatenate(feats)
    targets = np.concatenate(targets)
    gender_labels = np.concatenate(gender_labels)

    return feats, targets, gender_labels

def evaluate_gender(model, model_linear_target, model_linear_gender, dataloader):
    y_scores_target = []
    y_scores_gender = []
    y_true_target = []
    y_true_gender = []
    for i, (inputs, target, gender) in enumerate(dataloader):
        inputs, target, gender = inputs.to(device), target.float().to(device), gender.float().to(device)

        feat = model(inputs)
        pred_target = model_linear_target(feat).detach()
        pred_gender = model_linear_gender(feat).detach()

        y_scores_target.append(pred_target[:, 0].cpu().numpy())
        y_true_target.append(target.cpu().numpy())
        
        y_scores_gender.append(pred_gender[:, 0].cpu().numpy())
        y_true_gender.append(gender.cpu().numpy())

    y_scores_target = np.concatenate(y_scores_target)
    y_true_target = np.concatenate(y_true_target)
    y_scores_gender = np.concatenate(y_scores_gender)
    y_true_gender = np.concatenate(y_true_gender)
    
    d_00 = (y_true_gender==0)&(y_true_target==0) 
    d_01 = (y_true_gender==0)&(y_true_target==1) 
    d_10 = (y_true_gender==1)&(y_true_target==0) 
    d_11 = (y_true_gender==1)&(y_true_target==1) 
    
    cp = int(y_true_target.sum())
    y_scores_target_copy = np.copy(y_scores_target)
    y_scores_target_copy.sort()
    #print(cp)
    thresh = y_scores_target_copy[-cp]
    y_pred_target=np.where(y_scores_target>=thresh, 1, 0)

    ap_target = average_precision_score(y_true_target, y_scores_target)

    try:
        tp_target_00 = confusion_matrix(y_true_target[d_00], y_pred_target[d_00])[0][1] / len(y_pred_target[d_00])
        tp_target_01 = confusion_matrix(y_true_target[d_01], y_pred_target[d_01])[1][1] / len(y_pred_target[d_01])
        tp_target_10 = confusion_matrix(y_true_target[d_10], y_pred_target[d_10])[0][1] / len(y_pred_target[d_10])
        tp_target_11 = confusion_matrix(y_true_target[d_11], y_pred_target[d_11])[1][1] / len(y_pred_target[d_11])

    except:
        tp_target_00 = tp_target_01 = tp_target_10 = tp_target_11 = 0
    
    print("tp score:", tp_target_00, tp_target_01, tp_target_10, tp_target_11)
    EO_score = abs(tp_target_11 - tp_target_01) + abs(tp_target_10 - tp_target_00)
    print("EO_Score:", EO_score)

    ap_gender = average_precision_score(y_true_gender, y_scores_gender)
    
    return ap_target, ap_gender, EO_score


def evaluate_with_avg(model, model_linear_target, model_linear_gender, dataloader):
    y_scores_target = []
    y_scores_gender = []
    y_true_target = []
    y_true_gender = []

    avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    for i, (inputs, target, gender) in enumerate(dataloader):
        inputs, target, gender = inputs.to(device), target.float().to(device), gender.float().to(device)
        # print(inputs.shape, target.shape, gender.shape)
        feat = model(inputs)

        feat = avg(feat).view(-1, 512)
        pred_target = model_linear_target(feat).detach()
        pred_gender = model_linear_gender(feat).detach()

        y_scores_target.append(pred_target[:, 0].cpu().numpy())
        y_true_target.append(target.cpu().numpy())

        y_scores_gender.append(pred_gender[:, 0].cpu().numpy())
        y_true_gender.append(gender.cpu().numpy())

    y_scores_target = np.concatenate(y_scores_target)
    y_true_target = np.concatenate(y_true_target)
    y_scores_gender = np.concatenate(y_scores_gender)
    y_true_gender = np.concatenate(y_true_gender)

    d_00 = (y_true_gender == 0) & (y_true_target == 0)
    d_01 = (y_true_gender == 0) & (y_true_target == 1)
    d_10 = (y_true_gender == 1) & (y_true_target == 0)
    d_11 = (y_true_gender == 1) & (y_true_target == 1)

    cp = int(y_true_target.sum())
    y_scores_target_copy = np.copy(y_scores_target)
    y_scores_target_copy.sort()
    # print(cp)
    thresh = y_scores_target_copy[-cp]
    y_pred_target = np.where(y_scores_target >= thresh, 1, 0)

    ap_target = average_precision_score(y_true_target, y_scores_target)

    try:
        tp_target_00 = confusion_matrix(y_true_target[d_00], y_pred_target[d_00])[0][1] / len(y_pred_target[d_00])
        tp_target_01 = confusion_matrix(y_true_target[d_01], y_pred_target[d_01])[1][1] / len(y_pred_target[d_01])
        tp_target_10 = confusion_matrix(y_true_target[d_10], y_pred_target[d_10])[0][1] / len(y_pred_target[d_10])
        tp_target_11 = confusion_matrix(y_true_target[d_11], y_pred_target[d_11])[1][1] / len(y_pred_target[d_11])
    except:
        tp_target_00 = tp_target_01 = tp_target_10 = tp_target_11 = 0

    print("tp score:", tp_target_00, tp_target_01, tp_target_10, tp_target_11)
    EO_score = abs(tp_target_11 - tp_target_01) + abs(tp_target_10 - tp_target_00)
    print("EO_Score:", EO_score)

    ap_gender = average_precision_score(y_true_gender, y_scores_gender)

    return ap_target, ap_gender, EO_score


def evaluate_latent_codes(model, model_linear_target, model_linear_gender, dataloader):
    y_scores_target = []
    y_scores_gender = []
    y_true_target = []
    y_true_gender = []

    avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    for i, (inputs, target, gender) in enumerate(dataloader):
        inputs, target, gender = inputs.to(device), target.float().to(device), gender.float().to(device)
        # print(inputs.shape, target.shape, gender.shape)
        # feat = model(inputs)

        # feat = avg(feat).view(-1, 512)
        pred_target = model_linear_target(inputs).detach()
        pred_gender = model_linear_gender(inputs).detach()

        y_scores_target.append(pred_target[:, 0].cpu().numpy())
        y_true_target.append(target.cpu().numpy())

        y_scores_gender.append(pred_gender[:, 0].cpu().numpy())
        y_true_gender.append(gender.cpu().numpy())

    y_scores_target = np.concatenate(y_scores_target)
    y_scores_target_array = np.array_split(y_scores_target, 2)
    print("Distance:", np.linalg.norm(y_scores_target_array[0] - y_scores_target_array[1], 1), len(y_scores_target_array[0] - y_scores_target_array[1]))
    y_true_target = np.concatenate(y_true_target)
    y_scores_gender = np.concatenate(y_scores_gender)
    y_true_gender = np.concatenate(y_true_gender)

    d_00 = (y_true_gender == 0) & (y_true_target == 0)
    d_01 = (y_true_gender == 0) & (y_true_target == 1)
    d_10 = (y_true_gender == 1) & (y_true_target == 0)
    d_11 = (y_true_gender == 1) & (y_true_target == 1)

    cp = int(y_true_target.sum())
    y_scores_target_copy = np.copy(y_scores_target)
    y_scores_target_copy.sort()
    # print(cp)
    thresh = y_scores_target_copy[-cp]
    y_pred_target = np.where(y_scores_target >= thresh, 1, 0)

    ap_target = average_precision_score(y_true_target, y_scores_target)

    tp_target_00 = confusion_matrix(y_true_target[d_00], y_pred_target[d_00])[0][1] / len(y_pred_target[d_00])
    tp_target_01 = confusion_matrix(y_true_target[d_01], y_pred_target[d_01])[1][1] / len(y_pred_target[d_01])
    tp_target_10 = confusion_matrix(y_true_target[d_10], y_pred_target[d_10])[0][1] / len(y_pred_target[d_10])
    tp_target_11 = confusion_matrix(y_true_target[d_11], y_pred_target[d_11])[1][1] / len(y_pred_target[d_11])

    print("tp score:", tp_target_00, tp_target_01, tp_target_10, tp_target_11)
    EO_score = abs(tp_target_11 - tp_target_01) + abs(tp_target_10 - tp_target_00)
    print("EO_Score:", EO_score)

    ap_gender = average_precision_score(y_true_gender, y_scores_gender)

    return ap_target, ap_gender, EO_score


def BCELoss(pred, target):
    return -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))

def evaluate_raw(model, model_linear, dataloader):
    y_scores = []
    y_true = []
    for i, (inputs, target) in enumerate(dataloader):
        inputs, target = inputs.to(device), target.float().to(device)

        feat = model(inputs)
        pred = model_linear(feat).detach()

        y_scores.append(pred.cpu())
        y_true.append(target.cpu())

    y_scores = torch.cat(y_scores)
    y_true = torch.cat(y_true)
    ap = average_precision_score(y_true.numpy(), y_scores.numpy())
    return ap, y_scores, y_true


def evaluate_dp(model, model_linear, dataloader, dataloader_0, dataloader_1):
    
    model.eval()
    model_linear.eval()
    ap, _, _ = evaluate_raw(model, model_linear, dataloader)
    # calculate DP gap
    _, pred_0, y_test_0 = evaluate_raw(model, model_linear, dataloader_0)
    _, pred_1, y_test_1 = evaluate_raw(model, model_linear, dataloader_1)

    gap = pred_0.mean() - pred_1.mean()
    gap = abs(gap.cpu().numpy())
    
    y_test = torch.cat((y_test_0, y_test_1), 0)
    y_scores = torch.cat((pred_0, pred_1), 0)
    y_pred = y_scores > 0.5
    
    A_test = torch.cat((torch.zeros(pred_0.size(0)), torch.ones(pred_1.size(0))), 0)
    
    DP_05 = demographic_parity_difference(
        y_true=tensor_to_numpy_func(y_test),
        y_pred=tensor_to_numpy_func(y_pred),
        sensitive_features=tensor_to_numpy_func(A_test))
    # calculate average precision
    print("ap, gap, DP_05:", ap, gap, DP_05)
    return ap, gap, DP_05


def evaluate_dp2( model, model_linear, 
                dataloader,dataloader_0,dataloader_1):
    return 0 
    # model.eval()
    #
    # # calculate DP gap
    # # calculate average precision
    # # ap, _ = evaluate(model, model_linear, dataloader)
    #
    # _, pred_0,y_test_0 = evaluate_raw(model, model_linear, dataloader_0)
    # _, pred_1,y_test_1 = evaluate_raw(model, model_linear, dataloader_1)
    #
    # gap = pred_0.mean() - pred_1.mean()
    # gap = abs(gap.cpu().numpy())
    #
    # # calculate average precision
    #
    #
    #
    # y_scores_target = torch.cat([pred_0, pred_1], dim=0).cpu().numpy()
    # y_true_target = torch.cat([y_test_0, y_test_1],  dim=0).cpu().numpy()
    #
    # cp = int(y_true_target.sum())
    # y_scores_target_copy = np.copy(y_scores_target)
    # y_scores_target_copy.sort()
    # # print(cp)
    # thresh = y_scores_target_copy[-cp]
    # y_pred_0 = np.where(pred_0.cpu().numpy() >= thresh, 1, 0)
    # y_pred_1 = np.where(pred_1.cpu().numpy() >= thresh, 1, 0)
    # # try:
    # # tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    # try :
    #     tp_target_0 = confusion_matrix(y_test_0.squeeze(), y_pred_0.squeeze()).ravel()[-1] / pred_0.shape[0]
    #     tp_target_1 = confusion_matrix(y_test_1.squeeze(), y_pred_1.squeeze()).ravel()[-1] / pred_1.shape[0]
    # except :
    #     tp_target_0,tp_target_1 =0,0 
    #
    #
    #
    # # print("tp score:", tp_target_0, tp_target_1)
    # DP_score = abs(tp_target_1 - tp_target_0)
    # # print("DP_Score:", DP_score)
    #
    # return DP_score


def evaluate_dp3( model, model_linear, 
                dataloader,dataloader_0,dataloader_1):
    return 0 
    # model.eval()
    #
    # # calculate DP gap
    # # calculate average precision
    # # ap, _ = evaluate(model, model_linear, dataloader)
    #
    # _, pred_0,y_test_0 = evaluate_raw(model, model_linear, dataloader_0)
    # _, pred_1,y_test_1 = evaluate_raw(model, model_linear, dataloader_1)
    #
    # gap = pred_0.mean() - pred_1.mean()
    # gap = abs(gap.cpu().numpy())
    #
    # # calculate average precision
    #
    #
    #
    # y_scores_target = torch.cat([pred_0, pred_1], dim=0).cpu().numpy()
    # y_true_target = torch.cat([y_test_0, y_test_1],  dim=0).cpu().numpy()
    #
    # cp = int(y_true_target.sum())
    # y_scores_target_copy = np.copy(y_scores_target)
    # y_scores_target_copy.sort()
    # # print(cp)
    # thresh =0.5# y_scores_target_copy[-cp]
    # y_pred_0 = np.where(pred_0.cpu().numpy() >= thresh, 1, 0)
    # y_pred_1 = np.where(pred_1.cpu().numpy() >= thresh, 1, 0)
    # # try:
    # # tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    # try :
    #     tp_target_0 = confusion_matrix(y_test_0.squeeze(), y_pred_0.squeeze()).ravel()[-1] / pred_0.shape[0]
    #     tp_target_1 = confusion_matrix(y_test_1.squeeze(), y_pred_1.squeeze()).ravel()[-1] / pred_1.shape[0]
    #
    # except :
    #     tp_target_0,tp_target_1 = 0,0 
    #
    #
    # # print("tp score:", tp_target_0, tp_target_1)
    # DP_score = abs(tp_target_1 - tp_target_0)
    # # print("DP_Score:", DP_score)
    #
    # return DP_score




def evaluate_eo(model, model_linear, dataloader, dataloader_00, dataloader_01, dataloader_10, dataloader_11):
    model.eval()
    model_linear.eval()

    # calculate average precision
    ap, _, _ = evaluate_raw(model, model_linear, dataloader)

    _, pred_00, y_test_00 = evaluate_raw(model, model_linear, dataloader_00)
    _, pred_01, y_test_01 = evaluate_raw(model, model_linear, dataloader_01)
    _, pred_10, y_test_10 = evaluate_raw(model, model_linear, dataloader_10)
    _, pred_11, y_test_11 = evaluate_raw(model, model_linear, dataloader_11)

    gap_0 = pred_00.mean() - pred_10.mean()
    gap_1 = pred_01.mean() - pred_11.mean()
    gap_0 = abs(gap_0.cpu().numpy())
    gap_1 = abs(gap_1.cpu().numpy())

    gap = gap_0 + gap_1
    
    y_scores_target = torch.cat((pred_00, pred_01, pred_10, pred_11), dim=0).cpu().numpy()
    y_true_target = torch.cat([y_test_00, y_test_01, y_test_10, y_test_11], dim=0).cpu().numpy()
    
    cp = int(y_true_target.sum())
    y_scores_target_copy = np.copy(y_scores_target)
    y_scores_target_copy.sort()
    # print(cp)
    thresh =0.5# y_scores_target_copy[-cp]
    y_pred_00 = np.where(pred_00.cpu().numpy() >= thresh, 1, 0)
    y_pred_01 = np.where(pred_01.cpu().numpy() >= thresh, 1, 0)
    y_pred_10 = np.where(pred_10.cpu().numpy() >= thresh, 1, 0)
    y_pred_11 = np.where(pred_11.cpu().numpy() >= thresh, 1, 0)
    try:
        tp_target_00 = confusion_matrix(y_test_00.squeeze(), y_pred_00.squeeze()).ravel()[1] / pred_00.shape[0]
        tp_target_01 = confusion_matrix(y_test_01.squeeze(), y_pred_01.squeeze()).ravel()[-1] / pred_01.shape[0]
        tp_target_10 = confusion_matrix(y_test_10.squeeze(), y_pred_10.squeeze()).ravel()[1] / pred_10.shape[0]
        tp_target_11 = confusion_matrix(y_test_11.squeeze(), y_pred_11.squeeze()).ravel()[-1] / pred_11.shape[0]
    except:
        print("Warning:" + "*"*100)
        tp_target_00 = tp_target_01 = tp_target_10 = tp_target_11 = 0
    
    EO_score = abs(tp_target_11 - tp_target_01) + abs(tp_target_10 - tp_target_00)
    
    print("ap:", ap)
    print("gap:", gap, "score_00, score_01, score_10, score_11:", pred_00.mean().item(), pred_01.mean().item(),
          pred_10.mean().item(), pred_11.mean().item())
    print("EO_Score:", EO_score)

    return ap, gap, EO_score

def evaluate_eo2(model, model_linear, 
                dataloader, dataloader_00, dataloader_01, dataloader_10, dataloader_11,
                ):
    return 0 
    # model.eval()
    # model_linear.eval()
    #
    # # calculate average precision
    # # ap, _ = evaluate(model, model_linear, dataloader)
    #
    # _, pred_00,y_test_00 = evaluate_raw(model, model_linear, dataloader_00)
    # _, pred_01,y_test_01 = evaluate_raw(model, model_linear, dataloader_01)
    # _, pred_10,y_test_10 = evaluate_raw(model, model_linear, dataloader_10)
    # _, pred_11,y_test_11 = evaluate_raw(model, model_linear, dataloader_11)
    #
    #
    #
    # y_scores_target = torch.cat((pred_00, pred_01, pred_10, pred_11), dim=0).cpu().numpy()
    # y_true_target = torch.cat([y_test_00, y_test_01, y_test_10, y_test_11], dim=0).cpu().numpy()
    #
    # cp = int(y_true_target.sum())
    # y_scores_target_copy = np.copy(y_scores_target)
    # y_scores_target_copy.sort()
    # # print(cp)
    # thresh = y_scores_target_copy[-cp]
    # y_pred_00 = np.where(pred_00.cpu().numpy() >= thresh, 1, 0)
    # y_pred_01 = np.where(pred_01.cpu().numpy() >= thresh, 1, 0)
    # y_pred_10 = np.where(pred_10.cpu().numpy() >= thresh, 1, 0)
    # y_pred_11 = np.where(pred_11.cpu().numpy() >= thresh, 1, 0)
    # try:
    #     tp_target_00 = confusion_matrix(y_test_00.squeeze(), y_pred_00.squeeze()).ravel()[1] / pred_00.shape[0]
    #     tp_target_01 = confusion_matrix(y_test_01.squeeze(), y_pred_01.squeeze()).ravel()[-1] / pred_01.shape[0]
    #     tp_target_10 = confusion_matrix(y_test_10.squeeze(), y_pred_10.squeeze()).ravel()[1] / pred_10.shape[0]
    #     tp_target_11 = confusion_matrix(y_test_11.squeeze(), y_pred_11.squeeze()).ravel()[-1] / pred_11.shape[0]
    #
    # except:
    #     tp_target_00 = tp_target_01 = tp_target_10 = tp_target_11 = 0
    #
    # # print("tp score:", tp_target_00, tp_target_01, tp_target_10, tp_target_11)
    # EO_score = abs(tp_target_11 - tp_target_01) + abs(tp_target_10 - tp_target_00)
    # print("EO_Score:", EO_score)
    #
    # return EO_score





def evaluate_eo3(model, model_linear, 
                dataloader, dataloader_00, dataloader_01, dataloader_10, dataloader_11,
                ):
    return 0 
    # model.eval()
    # model_linear.eval()
    #
    # # calculate average precision
    # # ap, _ = evaluate(model, model_linear, dataloader)
    #
    # _, pred_00,y_test_00 = evaluate_raw(model, model_linear, dataloader_00)
    # _, pred_01,y_test_01 = evaluate_raw(model, model_linear, dataloader_01)
    # _, pred_10,y_test_10 = evaluate_raw(model, model_linear, dataloader_10)
    # _, pred_11,y_test_11 = evaluate_raw(model, model_linear, dataloader_11)
    #
    #
    #
    # y_scores_target = torch.cat((pred_00, pred_01, pred_10, pred_11), dim=0).cpu().numpy()
    # y_true_target = torch.cat([y_test_00, y_test_01, y_test_10, y_test_11], dim=0).cpu().numpy()
    #
    # cp = int(y_true_target.sum())
    # y_scores_target_copy = np.copy(y_scores_target)
    # y_scores_target_copy.sort()
    # # print(cp)
    # thresh =0.5# y_scores_target_copy[-cp]
    # y_pred_00 = np.where(pred_00.cpu().numpy() >= thresh, 1, 0)
    # y_pred_01 = np.where(pred_01.cpu().numpy() >= thresh, 1, 0)
    # y_pred_10 = np.where(pred_10.cpu().numpy() >= thresh, 1, 0)
    # y_pred_11 = np.where(pred_11.cpu().numpy() >= thresh, 1, 0)
    # try:
    #     tp_target_00 = confusion_matrix(y_test_00.squeeze(), y_pred_00.squeeze()).ravel()[1] / pred_00.shape[0]
    #     tp_target_01 = confusion_matrix(y_test_01.squeeze(), y_pred_01.squeeze()).ravel()[-1] / pred_01.shape[0]
    #     tp_target_10 = confusion_matrix(y_test_10.squeeze(), y_pred_10.squeeze()).ravel()[1] / pred_10.shape[0]
    #     tp_target_11 = confusion_matrix(y_test_11.squeeze(), y_pred_11.squeeze()).ravel()[-1] / pred_11.shape[0]
    # except:
    #     tp_target_00 = tp_target_01 = tp_target_10 = tp_target_11 = 0
    #
    # # except:
    # #     tp_target_00 = tp_target_01 = tp_target_10 = tp_target_11 = 0
    #
    # # print("tp score:", tp_target_00, tp_target_01, tp_target_10, tp_target_11)
    # EO_score = abs(tp_target_11 - tp_target_01) + abs(tp_target_10 - tp_target_00)
    # # print("EO_Score:", EO_score)
    #
    # return EO_score
