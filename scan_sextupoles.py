"""This file scans the effect of sextupoles on the DA."""

# oblanco 2025feb11

import at
import numpy as np

import multipole_variation

def set_mags(ring, idx, idxall, var, iorder):
    lenidxquarter = int(len(idxall)/4)
    quarter_ring = [0,1,2,3]
    # go on every quadrant
    for quadrant in quarter_ring:
        ele = idxall[idx + lenidxquarter*quadrant]
        if ring[ele].PolynomB[iorder] > 0:
            deltastep = +1*var
        else:
            deltastep = -1*var
        ring[ele].PolynomB[iorder] += deltastep


#ring = at.load_mat('ALBA_II_20250129_divideSH1SV2SV4SV7_OCT_B_withapertures.mat')
#ring = at.load_mat('ring_test2_chrom0.mat')
ring = at.load_mat('ALBA_II_20250129_divideSH1SV2SV4SV7_OCT_C_withapertures.mat')


idx_sh = at.get_refpts(ring, 'SH*')
idx_sv = at.get_refpts(ring, 'SV*')
idxsall= np.sort(np.concatenate((idx_sh, idx_sv)))
idx_oh = at.get_refpts(ring, 'OH*')
idx_ov = at.get_refpts(ring, 'OV*')
idxoall= np.sort(np.concatenate((idx_oh, idx_ov)))

#set_mags(ring, 0, idxsall, -20, 2)
# set_mags(ring, 0, idxsall, -20, 2)
# set_mags(ring, 9, idxsall, -20, 2)
# set_mags(ring, 0, idxsall, -20, 2)
# set_mags(ring,33, idxsall, -20, 2)
# set_mags(ring, 0, idxsall, -20, 2)
# set_mags(ring,27, idxsall, -20, 2)
# set_mags(ring,27, idxsall, -20, 2)
# set_mags(ring,43, idxsall, -20, 2)
# set_mags(ring, 9, idxsall, -20, 2)
# set_mags(ring, 5, idxsall, -20, 2)
# set_mags(ring, 0, idxoall, -2000, 3)
# set_mags(ring, 13, idxsall,-30, 2)
# set_mags(ring, 30, idxoall,-10000, 3)
# set_mags(ring, 1, idxoall,+10000, 3)
# set_mags(ring, 0, idxsall,-30, 2)
# set_mags(ring, 6, idxoall,+10000, 3)
# set_mags(ring, 0, idxsall,-30, 2)
# set_mags(ring, 9, idxoall,+10000, 3)
# set_mags(ring,13, idxsall,-30, 2)
# set_mags(ring, 7, idxsall,-20, 2)
# set_mags(ring,16, idxsall,-20, 2)
# set_mags(ring,13, idxsall,-20, 2)
# set_mags(ring,10, idxoall,-2000, 3)
# set_mags(ring, 2, idxoall,-2000, 3)
# set_mags(ring,11, idxoall,-2000, 3)
# set_mags(ring, 8, idxoall,-2000, 3)
# set_mags(ring, 4, idxoall,-2000, 3)
# set_mags(ring,11, idxoall,-2000, 3)
# print('edited ring')

uu = multipole_variation.magnet_mod(ring,100,var=-20, first_letter='S')
#uu = multipole_variation.magnet_mod(ring,100,var=+2000, first_letter='O')

#at.save_mat(ring,'test_da.mat')

