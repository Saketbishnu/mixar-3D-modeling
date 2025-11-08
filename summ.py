import json
s = json.load(open("outputs/summary.json"))
print("mesh, minmax_mse, unitsphere_mse, minmax_mae, unitsphere_mae")
for mesh, info in s.items():
    rmin = info['results']['minmax']['errors']
    runi = info['results']['unitsphere']['errors']
    print(mesh, rmin['mse_total'], runi['mse_total'], rmin['mae_total'], runi['mae_total'])
