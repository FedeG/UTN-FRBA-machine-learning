# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from tabulate import tabulate

data = pd.read_csv('./data.csv', delimiter='|')

# Mostrar la cantidad de celdas
print(f'size: {data.size}')
print(f'columns: {len(data.columns)}')
print(f'rows: {data.shape[0]}')

assert data.shape[0] == 238616

data_without_eof = data[(data.client_id != '(238615 rows affected)')]

assert data_without_eof.shape[0] == 238615

data_without_duplicated = data_without_eof.drop_duplicates(
    subset=['Month', 'client_id']
)

assert data_without_duplicated.shape[0] == 238615

# Obtener solo los de 9 meses
nine_mounths = data_without_duplicated.groupby(
    'client_id')['Month'].count().reset_index()
data_with_9_months_clients = data_without_duplicated.merge(
    nine_mounths[nine_mounths.Month == 9][['client_id']],
    how='inner',
    on='client_id',
)

assert data_with_9_months_clients.shape[0] == 229086

# Training window: 6 month (from 2018-11-01 to 2019-01-01)
# Lead window: 1 month (2019-02-01)
# Prediction window: last 2 month (2019-03-01 and 2019-04-01)

last_training_month = '2019-01-01'
last_training_month_data = data_with_9_months_clients[
    data_with_9_months_clients.Month == last_training_month]

print(last_training_month_data.CreditCard_CoBranding.value_counts())

data_without_cobranding = data_with_9_months_clients.merge(
    last_training_month_data[last_training_month_data.CreditCard_CoBranding == 'No'][[
        'client_id']],
    how='inner',
    on='client_id',
)

assert len(data_without_cobranding.client_id.unique()) == 22722

last_training_month_data = data_without_cobranding[
    data_without_cobranding.Month == last_training_month
]

print(last_training_month_data.Package_Active.value_counts())

data_package_active = data_without_cobranding.merge(
    last_training_month_data[last_training_month_data.Package_Active == 'No'][[
        'client_id']],
    how='inner',
    on='client_id',
)

assert len(data_package_active.client_id.unique()) == 22279

# identity_features: se mantiene igual en el abt
identity_features_columns = [
    'client_id', 'Month', 'First_product_dt', 'Last product_dt',
    'CreditCard Premium', 'CreditCard CoBranding', 'CreditCard_Active',
    'Loan_Active', 'Mortgage_Active', 'SavingAccount_Active_ARG_Salary',
    'SavingAccount Active ARG', 'SavingAccount_Active_DOLLAR',
    'DebitCard Active', 'Investment Active',
    'Insurance Life', 'Insurance Home', 'Insurance Accidents',
    'Insurance Mobile', 'Insurance ATM', 'Insurance Unemployment', 'Sex',
    'client_Age_grp', 'Mobile', 'Email',
    'Region', 'CreditCard_Product'
]
identity_features = data_package_active[
    data_package_active.Month == last_training_month
][identity_features_columns]

data_package_active.drop(identity_features_columns, axis=1, inplace=True)

# Transform Features: transformadas por logica
# client_Age_grp -> orden
transformDict = {
    'texto del transformDict': 0,
}
identity_features['client_Age_grp_ord'] = identity_features['client_Age_grp'].map(
    transformDict)

# sum Active Insurance
identity_features['Active Insurance'] = np.where(identity_features. Insurance_Life == 'Yes', 1, 0) + np.where(identity_features. Insurance Home == 'Yes', 1, 0)\
    np.where(identity_features. Insurance Accidents a 'Yes', 1, 0)\ np.where(identity_features. Insurance ATM == 'Yes', 1, 0)\ np.where(identity_features. Insurance Mobile 'Yes', 1, 0)\ np.where(identity_features. Insurance_Unemployment 'Yes', 1, 0)

# sum active products
identity_features['Active Products'] = np.where(identity_features. Loan Active == 'Yes', 1, 6)\ np.where(identity_features.CreditCard Active == 'Yes', 1, 0)\ np.where(identity_features.Mortgage Active == 'Yes', 1, 0)
+ np.where(identity_features. SavingAccount Active ARG == 'Yes', 1, 0)\ np.where(identity_features. SavingAccount Active DOLLAR == 'Yes', 1, 0)\ np.where(identity_features.Debit Card Active == 'Yes', 1, 0)\ + np.where(identity_features. Investment Active 'Yes', 1, 0) 1 + np.where(identity_features.Active Insurance > 0, 1, 0)

# video 1:47

# Aggregate Features: calculadas a partir de los datos que estan (nueva)

# check nulos
data.columns[data.isnull().any()].tolist()
