import unittest
import pandas as pd
from sklearn.base import ClassifierMixin
from train import ingest_data, clean_data, train_model

class TestTrain(unittest.TestCase):
    def setUp(self):
        self.file_path = 'train/titanic.xls'
        self.titanic = ingest_data(self.file_path)
        self.cleaned_titanic = clean_data(self.titanic)
        self.model = train_model(self.cleaned_titanic)

    def test_ingest_data(self):
        self.assertIsInstance(self.titanic, pd.DataFrame)
        self.assertFalse(self.titanic.empty)

    def test_clean_data(self):
        self.assertIsInstance(self.cleaned_titanic, pd.DataFrame)
        self.assertFalse(self.cleaned_titanic.empty)
        self.assertEqual(len(self.cleaned_titanic.columns), 4)
        self.assertTrue(all(col in self.cleaned_titanic.columns for col in ['survived', 'pclass', 'sex', 'age']))
        self.assertTrue(all(val in [0, 1] for val in self.cleaned_titanic['sex']))
        self.assertFalse(self.cleaned_titanic.isnull().values.any())

    def test_train_model(self):
        self.assertIsInstance(self.model, ClassifierMixin)

if __name__ == '__main__':
    unittest.main()