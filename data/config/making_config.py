import yaml
import numpy as np

# ============================================================
# 1. Učitaj originalni YAML
# ============================================================

input_file = 'config_apophis.yaml'

with open(input_file, "r") as f:
    config = yaml.safe_load(f)
# ============================================================
# 2. Promeni parametre
# ============================================================

TI = np.linspace(116, 1150, 10)
initialization = np.linspace(0.2, 2, 5)
# --------------------------------------------------
# Example
# --------------------------------------------------
for i in range(len(TI)):
    for j in range(len(initialization)):
        config["thermal_inertia"] = float(TI[i])
        config["orbital_initialisation"] = float(initialization[j])
        
        config["output_file"] = f'apophis_yarko_{i}_{j}.txt'
        config["progress_file"] = f'apophis_progress_{i}_{j}.txt'



        # ============================================================
        # 3. Sačuvaj kao novi YAML
        # ============================================================

        output_file = f"config_apophis_{i}_{j}.yaml"

        with open(output_file, "w") as f:
            yaml.safe_dump(
                config,
                f,
                sort_keys=False
            )


        print(f"Configuration saved to: {output_file}")