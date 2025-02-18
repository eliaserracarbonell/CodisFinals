"""This file is an example of parallel tracking"""

# oblanco 2025feb18
# based on this example
# https://stackoverflow.com/questions/47166239/python-3-create-new-process-when-another-one-finishes

import time
from multiprocessing import Pool

import at
import numpy as np

nparticles = 1000
ndims = 6
raw = np.zeros((ndims, nparticles))
tracked = np.zeros((nparticles), dtype=bool)
lost = np.zeros((nparticles), dtype=bool)

ring = at.load_mat("ring.mat")

nproc = 4
nturns = 200
if __name__ == "__main__":
    with Pool(processes=nproc) as p:
        nblocks = int(nparticles / nproc)
        for idxblock in range(nblocks):
            results = []
            for idxproc in range(nproc):
                idx = nproc * idxblock + idxproc
                r = raw[:, idx]
                result = p.apply_async(
                    at.lattice_track,
                    args=(ring, r),
                    kwds=dict(nturns=nturns, losses=True),
                )
                results.append(result)
                tracked[idxproc] = True
            print(f"Block {idxblock} of {nblocks} launched", end="")
            for r in results:
                r.wait()
            print(f" ... tracked")
            for idxproc in range(nproc):
                _, _, trackdata = results[idxproc].get()
                idx = nproc * idxblock + idxproc
                lost[idx] = trackdata["loss_map"]["islost"]
    p.close()
    p.join()
if np.all(tracked):
    print("all particles tracked")
print(f"Number of lost particles {lost.sum()}")
