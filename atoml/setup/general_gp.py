"""Function to setup a general GP."""
from .general_preprocess import GeneralPrepreprocess
from .general_kernel import general_kernel
from atoml.regression import GaussianProcess


class GeneralGaussianProcess(object):
    """Define a general setup for the Gaussin process.

    This should not be used to try and obtain highly accurate solutions. Though
    it should give a reasonable model.
    """

    def __init__(self, clean_type='eliminate', dimension='single'):
        """Initialize the class."""
        self.clean_type = clean_type
        self.dimension = dimension

    def train_gaussian_process(self, train_features, train_targets):
        """Generate a general gaussian process model."""
        train_features, train_targets = self._process_train_data(
            train_features, train_targets)

        kdict = general_kernel(train_features, self.dimension)

        self.gp = GaussianProcess(
            train_fp=train_features, train_target=train_targets,
            kernel_dict=kdict, regularization=1e-1,
            optimize_hyperparameters=True, scale_data=False
            )

        return self.gp

    def gaussian_process_predict(self, test_features):
        """Function to make GP predictions on tests data."""
        test_features = self.cleaner.transform(test_features)

        pred = self.gp.predict(test_fp=test_features)

        return pred

    def _process_train_data(self, train_features, train_targets):
        """Prepare the data."""
        self.cleaner = GeneralPrepreprocess(clean_type=self.clean_type)
        train_features, train_targets, _ = self.cleaner.process(
            train_features, train_targets)

        return train_features, train_targets
