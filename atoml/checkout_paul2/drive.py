"""Run the hierarchy."""
from cross_validation import HierarchyValidation
from pltfile import violinplot
from pltfile import originalplot


hv = HierarchyValidation(db_name='../../data/train_db.sqlite',
                         table='FingerVector',
                         file_name='hierarchey')

# If you want to keep datasize fixed, and vary features.
featselect_featvar = True
# If False above. Feature selection for ordinary ridge regression, or just
# plain vanilla ridge.
featselect_featconst = True
vio = False

i = 30
lim = i+4
subplot = 1
if featselect_featvar:
    # Under which interval of feature set in inspection.
    # (Varying of feature size)
    select_limit = [0, 2]
else:
    # For which fetureset inspection is made. (Varying of data size)
    select_limit = [i-1, i+1]
while i < lim:
    set_size, p_error, PC = hv.hierarcy(
       370, 999, 2000, hv, new_data=True,
       ridge=True, scale=True, globalscale=True, normalization=True,
       featselect_featvar=featselect_featvar,
       featselect_featconst=featselect_featconst,
       select_limit=select_limit, feat_sub=i)
    if not (set_size and p_error) is None:
        # Make four subplots.
        if vio:
            # Viloin subplot.
            violinplot(set_size, p_error, subplot, i)
        else:
            # Original subplot.
            originalplot(set_size, p_error, PC, subplot, i)
        i += 1
        subplot += 1
    else:
        print("No subset "+str(i))
        i += 1
        lim += 1
    select_limit = [i-1, i+1]
