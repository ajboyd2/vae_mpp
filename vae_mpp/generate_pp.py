from vae_mpp.parametric_pp import *
import numpy as np
import json

T = 100
configs = [
    [1.5, 0.0, 0.0],
    [0.0, 1.5, 0.0],
    [0.0, 0.0, 1.5],
    [1.0, 0.5, 0.0],
    [0.5, 1.0, 0.0],
    [0.0, 1.0, 0.5],
    [0.0, 0.5, 1.0],
    [1.0, 0.0, 0.5],
    [0.5, 0.0, 1.0],
]

configs.extend([(np.array(config)*2.5).tolist() for config in configs])
configs.extend([(np.array(config)*0.3).tolist() for config in configs])

'''    [0.5, 0.5, 0.5],
    [0.7, 0.5, 0.3],
    [0.7, 0.3, 0.5],
    [0.3, 0.7, 0.5],
    [0.3, 0.5, 0.7],
    [0.5, 0.7, 0.3],
    [0.5, 0.3, 0.7],
]'''

label_and_exs_per_pp = [
    ("train", 700),
    ("valid", 100),
    ("vis", 1),
]
pp = HomogenousPoissonProcess(K=3, scale=0)

for config in configs:
    pp.scale[:] = config
    user = "d_hpp_" + "_".join(str(i).replace(".", "") for i in list(config))
    print(user)

    for label, examples_per_pp in label_and_exs_per_pp:
        times, marks, _ = pp.generate_point_pattern(base_intensity=3, right_limit=T, batch_size=examples_per_pp)
        with open("/mnt/c/Users/Alex/Research/vae_mpp/data/hpp2/{}_{}.json".format(label, user), "w") as f:
            for time, mark in zip(times, marks):
                f.write(json.dumps({"user": user, "T": T, "times": list(time), "marks": list(int(m) for m in mark)}) + "\n")
        
        print("{} done".format(label))
