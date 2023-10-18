# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from tabulate import tabulate

data = pd.read_csv('./data.csv', delimiter='|', skipfooter=1, engine='python')

# Mostrar la cantidad de celdas
print(f'size: {data.size}')
print(f'columns: {len(data.columns)}')
print(f'rows: {data.shape[0]}')

assert len(data.client_id.unique()) == 26560

data_without_eof = data[(data.client_id != '(238615 rows affected)')]

assert len(data_without_eof.client_id.unique()) == 26560

data_without_duplicated = data_without_eof.drop_duplicates(
    subset=['Month', 'client_id']
)

assert len(data_without_duplicated.client_id.unique()) == 26560

# Obtener solo los de 9 meses
nine_mounths = data_without_duplicated.groupby(
    'client_id')['Month'].count().reset_index()
data_with_9_months_clients = data_without_duplicated.merge(
    nine_mounths[nine_mounths.Month == 9][['client_id']],
    how='inner',
    on='client_id',
)

assert len(data_with_9_months_clients.client_id.unique()) == 26483

# Training window: 6 month (from 2018-11-01 to 2019-01-01)
# Lead window: 1 month (2019-02-01)
# Prediction window: last 2 month (2019-03-01 and 2019-04-01)

last_training_month = '2019-01-01'
last_training_month_data = data_with_9_months_clients[
    data_with_9_months_clients.Month == last_training_month]

print(last_training_month_data.CreditCard_CoBranding.value_counts())

clients_without_cobranding = data_with_9_months_clients.merge(
    last_training_month_data[last_training_month_data.CreditCard_CoBranding == 'No'][[
        'client_id']],
    how='inner',
    on='client_id',
)

assert len(clients_without_cobranding.client_id.unique()) == 23646

last_training_month_data = clients_without_cobranding[
    clients_without_cobranding.Month == last_training_month
]

print(last_training_month_data.Package_Active.value_counts())

clients_without_cobranding_without_package = clients_without_cobranding.merge(
    last_training_month_data[last_training_month_data.Package_Active == 'No'][[
        'client_id']],
    how='inner',
    on='client_id',
)

assert len(clients_without_cobranding_without_package.client_id.unique()) == 23191

# identity_features: se mantiene igual en el abt
identity_features_columns = [
    'client_id', 'Month', 'First_product_dt', 'Last_product_dt',
    'CreditCard_Premium', 'CreditCard_CoBranding', 'CreditCard_Active',
    'Loan_Active', 'Mortgage_Active', 'SavingAccount_Active_ARG_Salary',
    'SavingAccount_Active_ARG', 'SavingAccount_Active_DOLLAR',
    'DebitCard_Active', 'Investment_Active',
    'Insurance_Life', 'Insurance_Home', 'Insurance_Accidents',
    'Insurance_Mobile', 'Insurance_ATM', 'Insurance_Unemployment', 'Sex',
    'Client_Age_grp', 'Mobile', 'Email',  'Region', 'CreditCard_Product'
]
identity_features = clients_without_cobranding_without_package[
    clients_without_cobranding_without_package.Month == last_training_month
][identity_features_columns]

# identity_features.columns[identity_features.isnull().any()].tolist()
# [Region, CreditCard_Product]
region_from_future = data[data.Month == '2019-03-01'].groupby(['Region', 'client_id']).size().reset_index()
identity_features.drop('Region', axis=1, inplace=True)
identity_features = identity_features.merge(region_from_future[['Region', 'client_id']], on='client_id', how='left');
identity_features['Region'] = identity_features['Region'].fillna('Missing')

identity_features['CreditCard_Product'] = identity_features['CreditCard_Product'].fillna('Missing')
identity_features['CreditCard_Product'] = np.where(identity_features.CreditCard_Active == 'No', 'No', identity_features.CreditCard_Product)

clients_without_cobranding_without_package.drop(identity_features_columns, axis=1, inplace=True)

# Transform Features: transformadas por logica
# client_Age_grp -> orden
transformDict = {
    'Menor a 18 años': 1800,
    'Entre 18 y 29 años': 1829,
    'Entre 30 y 39 años': 3039,
    'Entre 40 y 49 años': 4049,
    'Entre 50 y 59 años': 5059,
    'Entre 60 y 64 años': 6064,
    'Entre 65 y 69 años': 6569,
    'Mayor a 70 años': 7000,
}
identity_features['Client_Age_grp_ordinal'] = identity_features['Client_Age_grp'].map(transformDict)

# sum Active Insurance
insurance_fields = [
    'Insurance_Life',
    'Insurance_Home',
    'Insurance_Accidents',
    'Insurance_Mobile',
    'Insurance_ATM',
    'Insurance_Unemployment',
]
identity_features['Active_Insurances'] = identity_features[insurance_fields].apply(lambda row: sum(np.where(row == 'Yes', 1, 0)), axis=1)
identity_features['Active_Insurance'] = np.where(identity_features.Active_Insurances > 0, 'Ÿes', 'No')

# sum active products
products_fields = [
    'Loan_Active',
    'Mortgage_Active',
    'CreditCard_Active',
    'SavingAccount_Active_ARG',
    'SavingAccount_Active_DOLLAR',
    'DebitCard_Active',
    'Active_Insurance',
]
identity_features['Active_Products'] = identity_features[products_fields].apply(lambda row: sum(np.where(row == 'Yes', 1, 0)), axis=1)

# sum operations
operation_fields = [column for column in clients_without_cobranding_without_package.columns if 'Operations' in column]
clients_without_cobranding_without_package['Operations'] = clients_without_cobranding_without_package[operation_fields].apply(lambda row: sum(row), axis=1)
for operation_field in operation_fields:
    clients_without_cobranding_without_package[f'Operations_{operation_field}_vs_total'] = np.where(
        clients_without_cobranding_without_package['Operations'] == 0,
        np.floor(clients_without_cobranding_without_package['Operations']),
        np.floor(clients_without_cobranding_without_package[operation_field] / clients_without_cobranding_without_package['Operations'])
    )

"""
# sum transactions
transaction_fields = [column for column in clients_without_cobranding_without_package.columns if 'Transactions' in column and 'SavingAccount' in column]
clients_without_cobranding_without_package['Transactions'] = clients_without_cobranding_without_package[transaction_fields].apply(lambda row: sum(row), axis=1)
for transaction_field in transaction_fields:
    clients_without_cobranding_without_package[f'Transactions_{transaction_fields}_vs_total'] = np.where(
        clients_without_cobranding_without_package['Transactions'] == 0,
        clients_without_cobranding_without_package['Transactions'],
        (clients_without_cobranding_without_package[transaction_fields] / clients_without_cobranding_without_package['Transactions'])
    )

# sum payments
payment_fields = [column for column in clients_without_cobranding_without_package.columns if 'Payment' in column]
clients_without_cobranding_without_package['Payments'] = clients_without_cobranding_without_package[payment_fields].apply(lambda row: sum(row), axis=1)
for payment_field in payment_fields:
    clients_without_cobranding_without_package[f'Payments_{payment_fields}_vs_total'] = np.where(
        clients_without_cobranding_without_package['Payments'] == 0,
        clients_without_cobranding_without_package['Payments'],
        (clients_without_cobranding_without_package[payment_fields] / clients_without_cobranding_without_package['Payments'])
    )
"""

# Months between products
identity_features['First_product_dt'] = pd.to_datetime(identity_features['First_product_dt'])
identity_features['Last_product_dt'] = pd.to_datetime(identity_features['Last_product_dt'])
identity_features['Month'] = pd.to_datetime(identity_features['Month'])

identity_features['Last_first_product_Months'] = ((identity_features['Last_product_dt'] - identity_features['First_product_dt']).dt.days) / 30
identity_features['Month_first_product_Months'] = ((identity_features['Month'] - identity_features['First_product_dt']).dt.days) / 30
identity_features['Month_last_product_Months'] = ((identity_features['Month'] - identity_features['Last_product_dt']).dt.days) / 30

clients_without_cobranding_without_package['SavingAccount_Balance_Average'] = np.where(clients_without_cobranding_without_package.SavingAccount_Balance_Average.isnull(), (clients_without_cobranding_without_package.SavingAccount_Balance_FirstDate + clients_without_cobranding_without_package.SavingAccount_Balance_LastDate) / 2, clients_without_cobranding_without_package.SavingAccount_Balance_Average)

clients_without_cobranding_without_package['Card_Debit_vs_credit'] = np.where(
    clients_without_cobranding_without_package['SavingAccount_DebitCard_Spend_Amount'] == 0,
    clients_without_cobranding_without_package['CreditCard_Total_Spending'],
    (clients_without_cobranding_without_package['CreditCard_Total_Spending'] / clients_without_cobranding_without_package['SavingAccount_DebitCard_Spend_Amount'])
)

clients_without_cobranding_without_package['SavingAccount_Debit_vs_credit'] = np.where(
    clients_without_cobranding_without_package['SavingAccount_Debits_Amounts'] == 0,
    clients_without_cobranding_without_package['SavingAccount_Credits_Amounts'],
    (clients_without_cobranding_without_package['SavingAccount_Credits_Amounts'] / clients_without_cobranding_without_package['SavingAccount_Debits_Amounts'])
)

CER_VALUES = {
    '2018-08-01': 9.9316,
    '2018-09-01': 10.2663,
    '2018-10-01': 10.6234,
    '2018-11-01': 11.1948,
    '2018-12-01': 11.8454,
    '2019-01-01': 12.3512,
    '2019-02-01': 12.7058,
    '2019-03-01': 13.0390,
    '2019-04-01': 13.5000,
}

def calculate_with_cer(row, field):
    month = row['Month']
    cer_value = CER_VALUES.get(month)
    return row[field] / cer_value

# money
def process_money_field(data, field, min=0, sigmas=3):
    new_data = data.copy()

    # Remove min
    new_data = new_data[new_data[field] >= min]  

    # Remove outliers N sigmas
    sigma = sigmas * new_data[data[field] > min][field].std()
    new_data[field] = np.where(new_data[field] > sigma, sigma, new_data[field])

    # Add CER (Coeficiente de Estabilización de Referencia) value
    field_cer = f'{field}_CER'
    new_data[field_cer] = new_data.apply(calculate_with_cer, args=(field,), axis=1)

    # Add stats columns
    new_data = new_data.group_by(['client_id'])[[field, field_cer]].agg([
        np.sum, np.amax, np.min, np.mean, np.median, np.count_nonzero,
        np.unique, np.var
    ])

    return new_data

money_fields = [
    'SavingAccount_Balance_FirstDate',
    'SavingAccount_Balance_LastDate',
    'SavingAccount_Balance_Average',
    'SavingAccount_Salary_Payment_Amount',
    'SavingAccount_Transfer_In_Amount',
    'SavingAccount_ATM_Extraction_Amount',
    'SavingAccount_Service_Payment_Amount',
    'SavingAccount_CreditCard_Payment_Amount',
    'SavingAccount_Transfer_Out_Amount',
    'SavingAccount_DebitCard_Spend_Amount',
    'SavingAccount_Total_Amount',
    'SavingAccount_Credits_Amounts',
    'SavingAccount_Debits_Amounts',
    'CreditCard_Balance_ARG',
    'CreditCard_Total_Limit',
    'CreditCard_Total_Spending',
]
for money_field in money_fields:
    clients_without_cobranding_without_package = process_money_field(clients_without_cobranding_without_package, money_field, min=150)
