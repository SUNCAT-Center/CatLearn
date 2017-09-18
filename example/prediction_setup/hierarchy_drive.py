"""Run the hierarchy with feature selection."""
from atoml.cross_validation import Hierarchy
from atoml.learning_curve.pltfile import violinplot
from atoml.learning_curve.pltfile import originalplot
from atoml.learning_curve import hierarchy


hv = Hierarchy(db_name='../../data/train_db.sqlite', table='FingerVector',
               file_name='hierarchy')

# If you want to keep datasize fixed, and vary features.
featselect_featvar = False
# If False above. Feature selection for ordinary ridge regression, or just
# plain vanilla ridge.
featselect_featconst = True
vio = False

# i is the size of featureset.
i = 100
lim = i+4
subplot = 1
if featselect_featvar:
    # Under which interval of feature set in inspection.
    # (Varying of feature size)
    select_limit = [0, 20]
else:
    # For which fetureset inspection is made. (Varying of data size)
    select_limit = [i-1, i+1]
while i < lim:
    set_size, p_error, result, PC = hierarchy(
        hv, 370, 10, 50, new_data=True, ridge=True, scale=True,
        globalscale=True, normalization=True,
        featselect_featvar=featselect_featvar,
        featselect_featconst=featselect_featconst, select_limit=select_limit,
        feat_sub=i)
    if not (set_size and p_error) == [] and not featselect_featvar:
        for data in result:
            print('data size:', data[0], 'prediction error:',
                  data[1], 'Omega:', data[5],
                  'Euclidean length:', data[2],
                  'Pearson correlation:', data[3])
        # Make four subplots.
        if vio:
            # Viloin subplot.
            violinplot(set_size, p_error, subplot, i)
        else:
            # Original subplot.
            originalplot(set_size, p_error, PC, subplot, i)
        # Making the next subplot.
        i += 1
        subplot += 1
    elif (set_size and p_error) == [] and not featselect_featvar:
        print("No subset "+str(i))
        i += 1
        lim += 1
    if featselect_featvar:
        # Don't want to make four subpl for varying of features.
        i += 4
    select_limit = [i-1, i+1]
