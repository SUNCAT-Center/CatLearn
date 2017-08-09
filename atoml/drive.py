from cross_validation import HierarchyValidation

hv = HierarchyValidation(db_name='../data/train_db.sqlite',
                         table='FingerVector',
                         file_name='hierarchey')
hv.hierarcy(30, 10,p=30 ,new_data=True,
            ridge=True, scale=True, globalscale=True, normalization=True,
            feature_selection=True)
