# import numpy as np
# from pathlib import Path
#
# path = Path(r"./nn_cavity_spectral_features_20260620_233057.npz")
# data = np.load(path, allow_pickle=True)
#
# for k in data.files:
#     arr = data[k]
#     try:
#         print(f"{k:35s} shape={arr.shape}, dtype={arr.dtype}")
#     except Exception:
#         print(f"{k:35s} type={type(arr)}")
# print("spectra_norm_ds:", data["spectra_norm_ds"].shape, data["spectra_norm_ds"].dtype)
# print("wavelengths_spectra_saved_um:", data["wavelengths_spectra_saved_um"].shape)
# print("spectral_features_full:", data["spectral_features_full"].shape)
# print("spectral_feature_names:", data["spectral_feature_names"])
#
# for key in ["spectra_norm_ds", "spectral_features_full", "L_fft_um", "H_peak", "delta_L_nm"]:
#     arr = data[key]
#     print(key, "nan:", np.isnan(arr).sum(), "finite ratio:", np.isfinite(arr).mean())
#
#
# valid = data["valid_mask"].astype(bool)
# print("valid samples:", valid.sum())
# print("total samples:", len(valid))
# print("valid ratio:", valid.mean())
#
# process_id = data["process_id"]
# split_id = data["split_id"]
#
# for sid, name in enumerate(["train", "val", "test"]):
#     p = np.unique(process_id[split_id == sid])
#     print(name, "num processes:", len(p), "num samples:", np.sum(split_id == sid))
#
# train_p = set(process_id[split_id == 0])
# val_p = set(process_id[split_id == 1])
# test_p = set(process_id[split_id == 2])
#
# print("train ∩ val:", len(train_p & val_p))
# print("train ∩ test:", len(train_p & test_p))
# print("val ∩ test:", len(val_p & test_p))
#
# unique_p, counts = np.unique(process_id, return_counts=True)
# print("num processes:", len(unique_p))
# print("min samples/process:", counts.min())
# print("max samples/process:", counts.max())
# print("mean samples/process:", counts.mean())

import numpy as np
from pathlib import Path

path = Path(r"./nn_cavity_spectral_features_20260620_233057.npz")
data = np.load(path, allow_pickle=True)

X = data["spectral_features_full"]
names = data["spectral_feature_names"]

print("spectral_features_full shape:", X.shape)

for i, name in enumerate(names):
    col = X[:, i]
    nan_count = np.isnan(col).sum()
    finite_ratio = np.isfinite(col).mean()
    print(f"{i:02d} {name:28s} nan={nan_count:8d}, finite_ratio={finite_ratio:.6f}")