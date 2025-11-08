import argparse
import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

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
                    idx = p.split('/')[0]
                    try:
                        vi = int(idx)
                        if vi < 0:
                            vi = len(verts) + 1 + vi
                        face.append(vi-1)
                    except:
                        pass
                if len(face) >= 3:
                    faces.append(face[:3])
    return np.array(verts, dtype=float), np.array(faces, dtype=int)

def write_obj(path, verts, faces):
    # verts: Nx3 floats
    with open(path, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write("f " + " ".join(str(int(i)+1) for i in face) + "\n")

def minmax_normalize(v):
    vmin = v.min(axis=0)
    vmax = v.max(axis=0)
    denom = (vmax - vmin)
    denom[denom == 0] = 1.0
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

def zscore_normalize(v):
    mu = v.mean(axis=0)
    sigma = v.std(axis=0)
    sigma[sigma == 0] = 1.0
    norm = (v - mu) / sigma
    params = {'type':'zscore','mu':mu.tolist(),'sigma':sigma.tolist()}
    return norm, params

def quantize_norm(norm, bins=1024, norm_type='minmax'):
    nv = np.array(norm, dtype=float)
    if norm_type == 'unitsphere':
        nv_shift = (nv + 1.0) / 2.0
    elif norm_type == 'zscore':
        # For z-score, map to [0,1] based on robust min/max of normalized values
        # use min/max across data (clamp extreme outliers)
        lo = np.percentile(nv, 1)
        hi = np.percentile(nv, 99)
        nv_clip = np.clip(nv, lo, hi)
        nv_shift = (nv_clip - lo) / (hi - lo) if (hi - lo) != 0 else (nv_clip - lo)
    else:
        nv_shift = nv
    q = np.floor(np.clip(nv_shift, 0.0, 1.0) * (bins - 1)).astype(np.int32)
    return q

def dequantize(q, bins=1024, norm_type='minmax'):
    qf = q.astype(float) / (bins - 1)
    if norm_type == 'unitsphere':
        deq = qf * 2.0 - 1.0
    elif norm_type == 'zscore':
        # zscore dequantization is ambiguous (we stored clipped quantization);
        # we simply return qf in [0,1] and let denormalization step handle mapping with saved params
        deq = qf
    else:
        deq = qf
    return deq

def denormalize(deq, params):
    if params['type'] == 'minmax':
        vmin = np.array(params['vmin'])
        vmax = np.array(params['vmax'])
        recon = deq * (vmax - vmin) + vmin
        return recon
    elif params['type'] == 'unitsphere':
        centroid = np.array(params['centroid'])
        scale = float(params['scale'])
        recon = deq * scale + centroid
        return recon
    elif params['type'] == 'zscore':
        # For zscore, deq currently in [0,1] corresponding to clipped range; reconstruct using stored mu/sigma
        mu = np.array(params['mu'])
        sigma = np.array(params['sigma'])
        # Use min/max stored in params if present (we store them later). If not present, convert 0-1 -> mean via identity
        if 'z_lo' in params and 'z_hi' in params:
            lo = params['z_lo']
            hi = params['z_hi']
            z = deq * (hi - lo) + lo
        else:
            z = (deq - 0.5) * 2.0  # rough fallback
        recon = z * sigma + mu
        return recon
    else:
        raise RuntimeError("Unknown normalization type for denormalize")

def compute_errors(orig, recon):
    diff = orig - recon
    mse_per_axis = np.mean(diff**2, axis=0).tolist()
    mae_per_axis = np.mean(np.abs(diff), axis=0).tolist()
    mse_total = float(np.mean(np.sum(diff**2, axis=1)))
    mae_total = float(np.mean(np.sum(np.abs(diff), axis=1)))
    return {'mse_per_axis':mse_per_axis,'mae_per_axis':mae_per_axis,'mse_total':mse_total,'mae_total':mae_total}

def plot_error_bars(mse, mae, title, out_path):
    axes = ['X','Y','Z']
    x = np.arange(len(axes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(x - width/2, mse, width, label='MSE')
    ax.bar(x + width/2, mae, width, label='MAE')
    ax.set_xticks(x); ax.set_xticklabels(axes)
    ax.set_ylabel('Error'); ax.set_title(title)
    ax.legend(); fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def process_all(input_dir, output_dir, bins=1024, skip_visual=True, include_zscore=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    obj_paths = list(input_dir.rglob("*.obj")) + list(input_dir.rglob("*.ply"))
    if len(obj_paths) == 0:
        print("No .obj/.ply files found in", input_dir)
        return

    summary = {}
    for p in obj_paths:
        name = p.stem
        print(f"Processing: {name}")
        verts, faces = parse_obj(p)
        if verts.size == 0:
            print(f"  - No vertices parsed, skipping {name}")
            continue

        mesh_out_dir = output_dir / name
        ensure_dir(mesh_out_dir)

        # Task 1: stats
        stats = {'n_vertices': int(verts.shape[0]),
                 'min': verts.min(axis=0).tolist(),
                 'max': verts.max(axis=0).tolist(),
                 'mean': verts.mean(axis=0).tolist(),
                 'std': verts.std(axis=0).tolist()}
        (mesh_out_dir / f"{name}_stats.json").write_text(json.dumps(stats, indent=2))
        results = {}

        # Normalization methods to run
        methods = [('minmax', minmax_normalize), ('unitsphere', unit_sphere_normalize)]
        if include_zscore:
            methods.append(('zscore', zscore_normalize))

        for label, fn in methods:
            norm, params = fn(verts)

            # If zscore we compute robust clipping bounds and store them for denorm dequantization
            if label == 'zscore':
                lo = float(np.percentile(norm, 1))
                hi = float(np.percentile(norm, 99))
                if hi - lo == 0:
                    hi = lo + 1.0
                params['z_lo'] = lo
                params['z_hi'] = hi

            # Save normalized mesh (for visualization): normalized coordinates might be outside typical ranges,
            # but we still save them as floats
            norm_path = mesh_out_dir / f"{name}_normalized_{label}.obj"
            write_obj(norm_path, norm, faces)

            # Quantize -> integer array
            q = quantize_norm(norm, bins=bins, norm_type=label)
            np.save(mesh_out_dir / f"{name}_quantized_int_{label}.npy", q)

            # Also save a visual-friendly OBJ of quantized positions mapped to [0,1] floats
            q_float = q.astype(float) / (bins - 1)
            quant_obj_path = mesh_out_dir / f"{name}_quantized_vis_{label}.obj"
            write_obj(quant_obj_path, q_float, faces)

            # Dequantize & denormalize to reconstruct
            deq = dequantize(q, bins=bins, norm_type=label)
            recon = denormalize(deq, params)
            recon_path = mesh_out_dir / f"{name}_reconstructed_{label}.obj"
            write_obj(recon_path, recon, faces)

            # Compute errors
            errs = compute_errors(verts, recon)
            (mesh_out_dir / f"{name}_errors_{label}.json").write_text(json.dumps({'params': params, 'errors': errs}, indent=2))

            # Plot per-axis error
            plot_path = mesh_out_dir / f"{name}_error_{label}.png"
            plot_error_bars(errs['mse_per_axis'], errs['mae_per_axis'], f"Error - {name} - {label}", plot_path)

            # Collect results
            results[label] = {
                'params': params,
                'errors': errs,
                'norm_path': str(norm_path),
                'quant_int': str(mesh_out_dir / f"{name}_quantized_int_{label}.npy"),
                'quant_vis': str(quant_obj_path),
                'recon_path': str(recon_path),
                'plot': str(plot_path)
            }

        # Save small scatter of original vertices for quick visual check
        try:
            fig = plt.figure(figsize=(6,4))
            ax = fig.add_subplot(111, projection='3d')
            sample = verts if verts.shape[0] <= 5000 else verts[np.random.choice(verts.shape[0], 5000, replace=False)]
            ax.scatter(sample[:,0], sample[:,1], sample[:,2], s=1)
            ax.set_title(f"Original: {name}")
            orig_png = mesh_out_dir / f"{name}_original_scatter.png"
            fig.tight_layout()
            fig.savefig(orig_png)
            plt.close(fig)
        except Exception:
            # Matplotlib 3D could fail in headless systems; ignore if so
            pass

        summary[name] = {'stats': stats, 'results': results}

    # write global summary
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Create a small report template
    report_txt = output_dir / "report_template.txt"
    report_template = f"""
    Mesh Normalization, Quantization & Error Analysis
    Author: Saket Bishnu
    Files processed: {len(summary)} meshes

    Methods used:
    - Min-Max normalization + 1024-bin quantization
    - Unit Sphere normalization + 1024-bin quantization
    - (Optional) Z-score normalization (if enabled)

    For each mesh, the following are saved:
    - normalized mesh (.obj)
    - quantized integer array (.npy)
    - visual quantized mesh (.obj)
    - reconstructed mesh (.obj)
    - per-axis error plot (.png)
    - errors json (.json)

    Use summary.json to build tables of MSE/MAE and include screenshots/plots in the final PDF report.
    """
    report_txt.write_text(report_template)

    print("Processing complete. Outputs written to:", output_dir)
    print("Summary saved at:", summary_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesh Normalization, Quantization & Error Analysis")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder with .obj/.ply files (or nested folders)")
    parser.add_argument("--output_dir", type=str, required=True, help="Where outputs will be placed")
    parser.add_argument("--bins", type=int, default=1024, help="Quantization bins (e.g., 1024)")
    parser.add_argument("--no_zscore", action="store_true", help="Disable Z-score normalization method")
    args = parser.parse_args()

    process_all(args.input_dir, args.output_dir, bins=args.bins, include_zscore=(not args.no_zscore))
