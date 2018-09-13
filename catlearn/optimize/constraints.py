import numpy as np


def create_mask(ini, constraints):
    mask_constraints = np.ones_like(ini.positions, dtype=bool)
    len_contrains = len(constraints)
    for i in range(0, len_contrains):
        try:
            mask_constraints[constraints[i].__dict__['a']] = \
                                             ~(constraints[i].__dict__['mask'])
        except Exception:
            pass

        try:
            mask_constraints[constraints[i].__dict__['index']] = False
        except Exception:
            pass

        try:
            mask_constraints[constraints[0].__dict__['a']] = \
                                               ~constraints[0].__dict__['mask']
        except Exception:
            pass

        try:
            mask_constraints[constraints[-1].__dict__['a']] = \
                                            ~(constraints[-1].__dict__['mask'])
        except Exception:
            pass

    mask_constraints = mask_constraints.flatten()
    index_mask_constraints = np.argwhere(mask_constraints)
    return index_mask_constraints


def apply_mask(list_to_mask=None, mask_index=None):
    org_list_to_mask = list_to_mask
    masked_list = np.zeros((len(org_list_to_mask), len(mask_index)))
    for i in range(0, len(org_list_to_mask)):
        masked_list[i] = org_list_to_mask[i][mask_index].flatten()
    return [org_list_to_mask, masked_list]


def unmask_geometry(org_list, masked_geom, mask_index):
    if masked_geom.ndim == 0:
        masked_geom = np.asarray([[masked_geom]])
    if masked_geom.ndim == 1:
        masked_geom = [masked_geom]
    unmasked_geom = org_list[0].copy()
    for i in range(len(mask_index)):
        unmasked_geom[mask_index[i]] = masked_geom[0][i]
    return np.array(unmasked_geom)
