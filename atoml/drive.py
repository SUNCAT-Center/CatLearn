from cross_validation import HierarchyValidation

hv = HierarchyValidation(db_name='../data/train_db.sqlite',
                         table='FingerVector',
                         file_name='hierarchey')
hv.hierarcy(370, 10, 100, new_data=False,
            ridge=True, scale=True, globalscale=True, normalization=True)
