import numpy as np

out_path = (
    "~/einf220/fromDavid/gtsm_openDA_david/stochModel/input_dflowfm/steric_locs.xyn"
)
source_path = (
    "~/einf220/fromDavid/gtsm_openDA_david/stochModel/input_dflowfm/selected_output.xyn"
)

rng = np.random.default_rng(1234)
n_locs = 15

out_file = open(out_path, "w")
with open(source_path, "r") as file:
    n_lines = len(file.readlines())
    indices = rng.integers(0, n_lines, size=n_locs).tolist()
    for i, line in enumerate(file):
        if i in indices:
            out_file.write(line)
out_file.close()
