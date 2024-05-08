import pandas as pd    
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import ClassifierMixin
import joblib


def ingest_data(file_path: str) -> pd.DataFrame:
    """Reads data from a file and returns a DataFrame."""
    return pd.read_excel(file_path)

def clean_data(titanic: pd.DataFrame) -> pd.DataFrame:
    """Cleans the data and returns a DataFrame."""
    titanic = titanic.loc[:,('survived', 'pclass', 'sex', 'age')]
    titanic["sex"] = titanic['sex'].map({"female":0,'male':1})
    titanic.dropna(inplace=True)
    return titanic

def train_model(titanic: pd.DataFrame) -> ClassifierMixin:
    """Trains a model and returns it."""
    model = KNeighborsClassifier(3)
    X = titanic[['pclass','sex','age']]
    Y = titanic['survived']
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.2,random_state=42)
    model.fit(Xtrain,Ytrain)
    print(model.score(Xtest,Ytest))
    return model

def save_model(model: ClassifierMixin, file_path: str) -> None:
    """Saves a model to disk."""
    joblib.dump(model, file_path)

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
    titanic = ingest_data('train/titanic.xls')
    titanic = clean_data(titanic)
    model = train_model(titanic)
    save_model(model, 'model_titanic.joblib')
    print('Model trained and saved to disk.')

