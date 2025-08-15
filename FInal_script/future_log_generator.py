import os
from datetime import datetime

def make_artifact_folder(model_name, suffix, root="artifacts"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"{model_name}-{suffix}-{ts}"
    path = os.path.join(root, folder)
    os.makedirs(path, exist_ok=True)
    return path


# B. Add Metadata and Log File Initialization
# At the start of training (before the epoch loop)
# ...existing code...
artifact_root = "artifacts"  # or your preferred path
suffix = dp.upper()
artifact_dir = make_artifact_folder("CombinedRULModel", suffix, root=artifact_root)
log_path = os.path.join(artifact_dir, "train_val_log.txt")
meta_path = os.path.join(artifact_dir, "metadata.json")
ckpt_path = os.path.join(artifact_dir, "checkpoint.pth")

meta = {
    "model_name": "CombinedRULModel",
    "dp_mode": dp,
    "num_sensor_features": X_windows.shape[2],
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "num_epochs": NUM_EPOCHS,
    "privacy": dp,
    "dp_sigma": sigma if dp != "none" else None,
    "dp_clip_bound": MAX_GRAD_NORM if dp != "none" else None,
    "spec_k": spec_k if dp == "spectral" else None,
}
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=4)

with open(log_path, "w") as f:
    f.write("epoch,train_loss,val_loss,epoch_time,lr,notes\n")


# C. Log Each Epoch
# At the end of each epoch, after validation
elapsed = end - start
h, m, s = map(int, [elapsed//3600, (elapsed%3600)//60, elapsed%60])
et = f"{h:02d}:{m:02d}:{s:02d}"
notes = ""
if val_mse < best_val:
    best_val = val_mse
    no_improve = 0
    torch.save(model.state_dict(), ckpt_path)
    notes = f"Saved at epoch {epoch}"
else:
    no_improve += 1
    if no_improve % LR_PAT == 0:
        scheduler.step()
        notes += " LR stepped"
    if no_improve >= PATIENT:
        notes += " Early stopping"

with open(log_path, "a") as f:
    f.write(f"{epoch},{tloss:.6f},{val_mse:.6f},{et},{scheduler.get_last_lr()[0]:.6g},{notes.strip()}\n")
print(f"Epoch {epoch:02d} Train {tloss:.4f} Val {val_mse:.4f} Time {et} LR {scheduler.get_last_lr()[0]:.2e} {notes}")



# D. Update Metadata with Total Training Time
total = time.perf_counter() - start_all
h, m, s = map(int, [total//3600, (total%3600)//60, total%60])
tt = f"{h:02d}:{m:02d}:{s:02d}"

with open(meta_path, "r+") as f:
    d = json.load(f)
    d["total_training_time"] = tt
    f.seek(0)
    json.dump(d, f, indent=4)
    f.truncate()