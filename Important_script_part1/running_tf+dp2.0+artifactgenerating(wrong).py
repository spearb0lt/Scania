# combined_rul_pipeline.py

import os
import sys
import json
import time
import h5py
import joblib
import torch
import pandas as pd
import numpy as np

from datetime import datetime
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from tab_transformer_pytorch import TabTransformer
from torch.optim import Adam
from torch.nn import MSELoss
from sklearn.metrics import mean_squared_error, mean_absolute_error

# === Hard-coded paths ===
WINDOW_CSV     = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\Code\super_same_norm.csv"
SPEC_CSV       = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\data\train_specifications.csv"
ENCODER_PATH   = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\Important_script\spec_encoder.joblib"
H5_PATH        = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\Important_script\data_windows.h5"
ARTIFACT_ROOT  = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\Important_script\artifacts"
VALIDATION_CSV = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\Code\validation_super_same_norm.csv"
VALIDATION_H5  = r"C:\Users\ASUS\Desktop\SCANIA\2024-34-2\2024-34-2\Important_script\validation_data.h5"

SENSOR_FEATURES = [
    '171_0', '666_0', '427_0', '837_0', '167_0', '167_1', '167_2', '167_3', '167_4',
    '167_5', '167_6', '167_7', '167_8', '167_9', '309_0', '272_0', '272_1', '272_2',
    '272_3', '272_4', '272_5', '272_6', '272_7', '272_8', '272_9', '835_0', '370_0',
    '291_0', '291_1', '291_2', '291_3', '291_4', '291_5', '291_6', '291_7', '291_8',
    '291_9', '291_10', '158_0', '158_1', '158_2', '158_3', '158_4', '158_5', '158_6',
    '158_7', '158_8', '158_9', '100_0', '459_0', '459_1', '459_2', '459_3', '459_4',
    '459_5', '459_6', '459_7', '459_8', '459_9', '459_10', '459_11', '459_12', '459_13',
    '459_14', '459_15', '459_16', '459_17', '459_18', '459_19', '397_0', '397_1', '397_2',
    '397_3', '397_4', '397_5', '397_6', '397_7', '397_8', '397_9', '397_10', '397_11',
    '397_12', '397_13', '397_14', '397_15', '397_16', '397_17', '397_18', '397_19',
    '397_20', '397_21', '397_22', '397_23', '397_24', '397_25', '397_26', '397_27',
    '397_28', '397_29', '397_30', '397_31', '397_32', '397_33', '397_34', '397_35'
]

# === Utils ===
def create_X_y(csv_path=WINDOW_CSV, sensor_features=None, context=70, verbose=True):
    df = pd.read_csv(csv_path)
    X,y,vids = [],[],[]
    for vid,grp in df.groupby("vehicle_id"):
        data = grp[sensor_features].values
        rul  = grp["RUL"].values
        if len(data)<context:
            if verbose: print(f"Skipping {vid}, len<{context}")
            continue
        for i in range(len(data)-context+1):
            X.append(data[i:i+context])
            y.append(rul[i+context-1])
            vids.append(vid)
    X=np.stack(X); y=np.array(y); vids=np.array(vids)
    if verbose: print(f"Windows: {len(X)}, shape={X.shape[1:]}")
    spec_df = pd.read_csv(SPEC_CSV)
    spec_cols=[f"Spec_{i}" for i in range(8)]
    enc=OrdinalEncoder()
    spec_df[spec_cols]=enc.fit_transform(spec_df[spec_cols])
    specs = (
        pd.DataFrame({"vehicle_id":vids})
          .merge(spec_df[["vehicle_id"]+spec_cols],on="vehicle_id")
    )[spec_cols].values.astype(int)
    joblib.dump(enc, ENCODER_PATH)
    if verbose: print(f"Saved encoder→{ENCODER_PATH}")
    return X,y,vids,specs

def save_to_h5(X,y,vids,specs,h5_path=H5_PATH):
    with h5py.File(h5_path,"w") as f:
        f.create_dataset("X_windows",data=X,compression="gzip")
        f.create_dataset("y_labels", data=y,compression="gzip")
        f.create_dataset("window_vids",data=vids,compression="gzip")
        f.create_dataset("specs_per_window",data=specs,compression="gzip")
    print(f"Saved H5→{h5_path}")

def load_from_h5(h5_path=H5_PATH):
    with h5py.File(h5_path,"r") as f:
        X=f["X_windows"][:]
        y=f["y_labels"][:]
        vids=f["window_vids"][:]
        specs=f["specs_per_window"][:]
    return X,y,vids,specs

class RULCombinedDataset(Dataset):
    def __init__(self,X,specs,y):
        self.X=X; self.specs=specs; self.y=y.reshape(-1,1)
    def __len__(self): return len(self.y)
    def __getitem__(self,i):
        return (
            torch.from_numpy(self.specs[i]).long(),
            torch.from_numpy(self.X[i]).float(),
            torch.from_numpy(self.y[i]).float()
        )

def make_artifact_folder(model_name,pvt):
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix="DP" if pvt else "NDP"
    folder=f"{model_name}-{suffix}-{ts}"
    path=os.path.join(ARTIFACT_ROOT,folder)
    os.makedirs(path,exist_ok=True)
    return path

# === Models ===
class TimeSeriesEmbedder(nn.Module):
    def __init__(self,num_features,d_model=128,n_heads=8,num_layers=2,dropout=0.1):
        super().__init__()
        self.input_proj=nn.Linear(num_features,d_model)
        enc_layer=nn.TransformerEncoderLayer(
            d_model=d_model,nhead=n_heads,dropout=dropout,batch_first=True
        )
        self.encoder=nn.TransformerEncoder(enc_layer,num_layers=num_layers)
    def forward(self,x):
        x=self.input_proj(x)
        x=self.encoder(x)
        return x[:,-1,:]

class CombinedRULModel(nn.Module):
    def __init__(self,num_sensor_features,context_length,categories,continuous_dim,cont_mean_std=None):
        super().__init__()
        self.tf=TimeSeriesEmbedder(num_sensor_features,continuous_dim)
        if cont_mean_std is None:
            cont_mean_std=torch.stack([torch.zeros(continuous_dim),torch.ones(continuous_dim)],dim=1)
        self.tab=TabTransformer(
            categories=categories,
            num_continuous=continuous_dim,
            dim=continuous_dim,
            dim_out=1,
            depth=6,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            mlp_hidden_mults=(4,2),
            mlp_act=nn.ReLU(),
            continuous_mean_std=cont_mean_std
        )
    def forward(self,x_cat,x_ts):
        cont=self.tf(x_ts)
        return self.tab(x_cat,cont)

# === Services ===
def get_criterion(): return MSELoss()
def get_optimizer(model,lr=1e-3): return Adam(model.parameters(),lr=lr)

# === DP helper ===
def train_dp_batch(model,criterion,optimizer,x_cat,x_ts,yb,max_grad_norm,noise_multiplier):
    summed={n:torch.zeros_like(p) for n,p in model.named_parameters()}
    for i in range(x_cat.size(0)):
        xi_cat, xi_ts, yi = x_cat[i:i+1], x_ts[i:i+1], yb[i:i+1]
        loss=criterion(model(xi_cat,xi_ts),yi)
        grads=torch.autograd.grad(loss,model.parameters())
        total_norm=torch.sqrt(sum(g.norm()**2 for g in grads))
        clip_coef=(max_grad_norm/(total_norm+1e-6)).clamp(max=1.0)
        for (n,_),g in zip(model.named_parameters(),grads):
            summed[n]+=g*clip_coef
    for n,p in model.named_parameters():
        noise=torch.randn_like(summed[n])*(noise_multiplier*max_grad_norm)
        p.grad=(summed[n]+noise)/x_cat.size(0)

# === Training function ===
def train(use_h5=False,pvt=False):
    # hyperparams
    DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE=4; NUM_EPOCHS=5; LR=1e-3
    LR_PAT=5; ES_PAT=11; LR_F=0.5
    MAX_G=1.0; NOISE_M=1.0
    # data
    if use_h5 and os.path.exists(H5_PATH):
        X,y,_,specs=load_from_h5()
    else:
        X,y,_,specs=create_X_y(sensor_features=SENSOR_FEATURES)
        save_to_h5(X,y,_,specs)
    encoder=joblib.load(ENCODER_PATH)
    cat_sizes=tuple(len(c) for c in encoder.categories_)
    Xtr,Xv,str_,sv,yr,yv=train_test_split(X,specs,y,test_size=0.2,random_state=42)
    tl=DataLoader(RULCombinedDataset(Xtr,str_,yr),batch_size=BATCH_SIZE,shuffle=True)
    vl=DataLoader(RULCombinedDataset(Xv,sv,yv),batch_size=BATCH_SIZE,shuffle=False)
    # artifacts
    art=make_artifact_folder("CombinedRULModel",pvt)
    logp=os.path.join(art,"train_val_log.txt")
    metap=os.path.join(art,"metadata.json")
    ckpt=os.path.join(art,"checkpoint.pth")
    # metadata init
    meta={
        "model_name":"CombinedRULModel",
        "num_sensor_features":X.shape[2],
        "context_length":X.shape[1],
        "continuous_dim":128,
        "categories":list(cat_sizes),
        "batch_size":BATCH_SIZE,
        "learning_rate":LR,
        "num_epochs":NUM_EPOCHS,
        "pvt":pvt
    }
    with open(metap,"w") as f: json.dump(meta,f,indent=4)
    # model
    model=CombinedRULModel(X.shape[2],X.shape[1],cat_sizes,128).to(DEVICE)
    crit=get_criterion(); opt=get_optimizer(model,lr=LR)
    sched=StepLR(opt,step_size=1,gamma=LR_F)
    best=1e9; noimp=0
    with open(logp,"w") as f:
        f.write("epoch,train_loss,val_loss,epoch_time,lr,notes\n")
    start_all=time.perf_counter()
    for ep in range(1,NUM_EPOCHS+1):
        ep_start=time.perf_counter()
        lr_cur=opt.param_groups[0]['lr']
        # train
        model.train(); tloss=0
        for xc,xt,yb in tl:
            xc,xt,yb=xc.to(DEVICE),xt.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad()
            if pvt: train_dp_batch(model,crit,opt,xc,xt,yb,MAX_G,NOISE_M)
            l=crit(model(xc,xt),yb); l.backward(); opt.step()
            tloss+=l.item()*yb.size(0)
        tloss/=len(tl.dataset)
        # val
        model.eval(); vloss=0
        with torch.no_grad():
            for xc,xt,yb in vl:
                xc,xt,yb=xc.to(DEVICE),xt.to(DEVICE),yb.to(DEVICE)
                l=crit(model(xc,xt),yb); vloss+=l.item()*yb.size(0)
        vloss/=len(vl.dataset)
        # timing
        ep_end=time.perf_counter(); elapsed=ep_end-ep_start
        h,m,s=map(int, [elapsed//3600, (elapsed%3600)//60, elapsed%60 ])
        et=f"{h:02d}:{m:02d}:{s:02d}"
        notes=""
        if vloss<best:
            best=vloss; noimp=0; torch.save(model.state_dict(),ckpt)
            notes=f"Model saved at epoch {ep}"
        else:
            noimp+=1
            if noimp%LR_PAT==0:
                sched.step(); notes+=" LR stepped"
            if noimp>=ES_PAT:
                notes+=" Early stopping"
        with open(logp,"a") as f:
            f.write(f"{ep},{tloss:.6f},{vloss:.6f},{et},{lr_cur:.6g},{notes.strip()}\n")
        print(f"Epoch {ep:02d} Train {tloss:.4f} Val {vloss:.4f} Time {et} LR {lr_cur:.2e} {notes}")
        if noimp>=ES_PAT:
            print("Early stop"); break
    total=time.perf_counter()-start_all
    h,m,s=map(int,[total//3600,(total%3600)//60,total%60])
    tt=f"{h:02d}:{m:02d}:{s:02d}"
    # update metadata
    with open(metap,"r+") as f:
        d=json.load(f); d["total_training_time"]=tt
        f.seek(0); json.dump(d,f,indent=4); f.truncate()
    print(f"Done. Best val MSE {best:.4f}. Total time {tt}")

# I AM STARTING HERE :)
if __name__=="__main__":
   
    train(use_h5=True, pvt=False)