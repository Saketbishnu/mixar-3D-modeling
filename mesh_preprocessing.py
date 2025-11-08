# Running a self-contained mesh preprocessing pipeline on uploaded sample zip
# This code will:
# - Unzip /mnt/data/8samples.zip (uploaded by user)
# - Find .obj files inside
# - For each .obj: parse vertices and faces (simple parser)
# - Compute stats, apply Min-Max and Unit Sphere normalization
# - Quantize (bins=1024), dequantize, denormalize, reconstruct
# - Compute MSE and MAE per axis and total
# - Save reconstructed .obj files and plots to /mnt/data/outputs/
# - Print summary and list of generated files with download links

import os, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# === Saket's Local Setup ===
# Folder where your .obj files are stored:
WORK_DIR = Path(r"C:\Users\saket\OneDrive\Desktop\mesh_preprocessing\data\8samples")

# Folder where output results will be saved:
OUT_DIR = Path(r"C:\Users\saket\OneDrive\Desktop\mesh_preprocessing\outputs")

# Number of quantization bins:
BINS = 1024

# Ensure output folder exists
os.makedirs(OUT_DIR, exist_ok=True)



os.makedirs(OUT_DIR, exist_ok=True)



# 2. Find .obj files
obj_paths = list(Path(WORK_DIR).rglob("*.obj"))
if len(obj_paths) == 0:
    print("No .obj files found inside the uploaded zip. Contents of the zip:")
    for p in Path(WORK_DIR).rglob("*"):
        print(str(p.relative_to(WORK_DIR)))
    raise SystemExit("No .obj files to process. Please upload .obj meshes.")

print(f"Found {len(obj_paths)} .obj files. Processing...")

def parse_obj(path):
    verts = []
    faces = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                if len(parts) >= 4:
                    x,y,z = float(parts[1]), float(parts[2]), float(parts[3])
                    verts.append([x,y,z])
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                face = []
                for p in parts:
                    # face entries can be like v, v/vt, v//vn, v/vt/vn
                    idx = p.split('/')[0]
                    try:
                        vi = int(idx)
                        if vi < 0:
                            # negative indices refer to relative indexing
                            vi = len(verts) + 1 + vi
                        face.append(vi-1)  # convert to 0-based
                    except:
                        pass
                if len(face) >= 3:
                    # only keep triangles or first three verts
                    faces.append(face[:3])
    return np.array(verts, dtype=float), np.array(faces, dtype=int)

def write_obj(path, verts, faces):
    with open(path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            # write 1-based indices
            f.write("f " + " ".join(str(int(i)+1) for i in face) + "\n")

def minmax_normalize(v):
    vmin = v.min(axis=0)
    vmax = v.max(axis=0)
    denom = vmax - vmin
    denom[denom==0] = 1.0
    norm = (v - vmin) / denom
    params = {'type':'minmax','vmin':vmin.tolist(),'vmax':vmax.tolist()}
    return norm, params

def unit_sphere_normalize(v):
    centroid = v.mean(axis=0)
    centered = v - centroid
    max_dist = np.max(np.linalg.norm(centered, axis=1))
    if max_dist == 0:
        max_dist = 1.0
    norm = centered / max_dist
    params = {'type':'unitsphere','centroid':centroid.tolist(),'scale':float(max_dist)}
    return norm, params

def quantize_norm(norm, bins=BINS, norm_type='minmax'):
    nv = np.array(norm, dtype=float)
    if norm_type == 'unitsphere':
        nv_shift = (nv + 1.0)/2.0  # map [-1,1] -> [0,1]
    else:
        nv_shift = nv  # assume in [0,1]
    q = np.floor(nv_shift * (bins - 1)).astype(np.int32)
    return q

def dequantize(q, bins=BINS, norm_type='minmax'):
    qf = q.astype(float) / (bins - 1)
    if norm_type == 'unitsphere':
        deq = qf * 2.0 - 1.0
    else:
        deq = qf
    return deq

def denormalize(deq, params):
    if params['type'] == 'minmax':
        vmin = np.array(params['vmin'])
        vmax = np.array(params['vmax'])
        recon = deq * (vmax - vmin) + vmin
        return recon
    else:
        centroid = np.array(params['centroid'])
        scale = float(params['scale'])
        recon = deq * scale + centroid
        return recon

def compute_errors(orig, recon):
    diff = orig - recon
    mse_per_axis = np.mean(diff**2, axis=0).tolist()
    mae_per_axis = np.mean(np.abs(diff), axis=0).tolist()
    mse_total = float(np.mean(np.sum(diff**2, axis=1)))
    mae_total = float(np.mean(np.sum(np.abs(diff), axis=1)))
    return {'mse_per_axis':mse_per_axis,'mae_per_axis':mae_per_axis,'mse_total':mse_total,'mae_total':mae_total}

generated_files = []

summary = {}
for path in obj_paths:
    name = Path(path).stem
    verts, faces = parse_obj(path)
    if verts.size == 0:
        print(f"Skipping {name}: no vertices parsed.")
        continue
    mesh_out_dir = Path(OUT_DIR)/name
    mesh_out_dir.mkdir(parents=True, exist_ok=True)
    # stats
    stats = {
        'n_vertices': int(verts.shape[0]),
        'min': verts.min(axis=0).tolist(),
        'max': verts.max(axis=0).tolist(),
        'mean': verts.mean(axis=0).tolist(),
        'std': verts.std(axis=0).tolist()
    }
    with open(mesh_out_dir / f"{name}_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    generated_files.append(str(mesh_out_dir / f"{name}_stats.json"))
    summary[name] = {'stats': stats, 'results': {}}
    # for both normalizations
    for norm_fn,label in [(minmax_normalize,'minmax'),(unit_sphere_normalize,'unitsphere')]:
        norm, params = norm_fn(verts)
        # save normalized as obj
        norm_path = mesh_out_dir / f"{name}_normalized_{label}.obj"
        write_obj(norm_path, norm, faces)
        generated_files.append(str(norm_path))
        # quantize
        q = quantize_norm(norm, bins=BINS, norm_type=label)
        # dequantize
        deq = dequantize(q, bins=BINS, norm_type=label)
        # denormalize/reconstruct
        recon = denormalize(deq, params)
        recon_path = mesh_out_dir / f"{name}_reconstructed_{label}.obj"
        write_obj(recon_path, recon, faces)
        generated_files.append(str(recon_path))
        # compute errors
        errs = compute_errors(verts, recon)
        with open(mesh_out_dir / f"{name}_errors_{label}.json", 'w') as f:
            json.dump({'params':params,'errors':errs}, f, indent=2)
        generated_files.append(str(mesh_out_dir / f"{name}_errors_{label}.json"))
        # plot per-axis errors
        axes = ['X','Y','Z']
        mse = errs['mse_per_axis']
        mae = errs['mae_per_axis']
        fig, ax = plt.subplots(figsize=(6,4))
        x = np.arange(len(axes))
        width = 0.35
        ax.bar(x - width/2, mse, width, label='MSE')
        ax.bar(x + width/2, mae, width, label='MAE')
        ax.set_xticks(x); ax.set_xticklabels(axes)
        ax.set_ylabel('Error'); ax.set_title(f"Error - {name} - {label}")
        ax.legend(); fig.tight_layout()
        plot_path = mesh_out_dir / f"{name}_error_{label}.png"
        fig.savefig(plot_path)
        plt.close(fig)
        generated_files.append(str(plot_path))
        # update summary
        summary[name]['results'][label] = {'params': params, 'errors': errs, 'norm_path': str(norm_path), 'recon_path': str(recon_path), 'plot': str(plot_path)}
    # Save a small view snapshot of original vertices (scatter plot)
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111, projection='3d')
    sample = verts if verts.shape[0] <= 5000 else verts[np.random.choice(verts.shape[0], 5000, replace=False)]
    ax.scatter(sample[:,0], sample[:,1], sample[:,2], s=1)
    ax.set_title(f"Original: {name}")
    plt.tight_layout()
    orig_png = mesh_out_dir / f"{name}_original_scatter.png"
    fig.savefig(orig_png)
    plt.close(fig)
    generated_files.append(str(orig_png))

# Save summary json
summary_path = Path(OUT_DIR)/"summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
generated_files.append(str(summary_path))

print("Processing complete. Generated files:")
for p in generated_files:
    print(p)

# Print download links (sandbox path)
print("\nDownload links (click to download):")
for p in generated_files:
    rel = os.path.relpath(p, "/mnt/data")
    print(f"[Download] sandbox:/mnt/data/{rel}")
