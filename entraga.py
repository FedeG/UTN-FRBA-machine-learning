# Import

import pandas as pd
import numpy as np

# Utils
def show_data_summary(data):
    print(f'Size: {data.size}')
    print(f'Columns: {len(data.columns)}')
    print(f'Rows: {data.shape[0]}')
    print(f'Unique clients: {len(data.client_id.unique())}')

# Load data
data = pd.read_csv('./data.csv', delimiter='|', skipfooter=1, engine='python')

show_data_summary(data)

# Remove summary row
data = data[(data.client_id != '(238615 rows affected)')]

show_data_summary(data)


# Remove duplicates
data = data.drop_duplicates(subset=['Month', 'client_id'])

show_data_summary(data)

# Get clients with 9 months data
nine_mouths = data.groupby('client_id')['Month'].count().reset_index()
clients_with_9_months = data.merge(
    nine_mouths[nine_mouths.Month == 9][['client_id']],
    how='inner',
    on='client_id',
)

show_data_summary(clients_with_9_months)

# Get last training month clients without cobranding
last_training_month = '2019-01-01'
last_training_month_data = clients_with_9_months[clients_with_9_months.Month == last_training_month]

clients_without_cobranding = clients_with_9_months.merge(
    last_training_month_data[last_training_month_data.CreditCard_CoBranding == 'No'][['client_id']],
    how='inner',
    on='client_id',
)

show_data_summary(clients_without_cobranding)

# Get last training month clients without package active
last_training_month_data = clients_without_cobranding[
    clients_without_cobranding.Month == last_training_month]

clients_without_cobranding_without_package = clients_without_cobranding.merge(
    last_training_month_data[last_training_month_data.Package_Active == 'No'][['client_id']],
    how='inner',
    on='client_id',
)

show_data_summary(clients_without_cobranding_without_package)

# Identity features
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

# Remove nulls
# identity_features.columns[identity_features.isnull().any()].tolist() -> [Region, CreditCard_Product]

# Remove nulls of Region
region_from_future = data[data.Month == '2019-03-01'].groupby(['Region', 'client_id']).size().reset_index()
identity_features.drop('Region', axis=1, inplace=True)
identity_features = identity_features.merge(region_from_future[['Region', 'client_id']], on='client_id', how='left')
identity_features['Region'] = identity_features['Region'].fillna('Missing')

# Remove nulls of CreditCard_Product
identity_features['CreditCard_Product'] = identity_features['CreditCard_Product'].fillna('Missing')
identity_features['CreditCard_Product'] = np.where(identity_features.CreditCard_Active == 'No', 'No', identity_features.CreditCard_Product)

# Add values to nulls of SavingAccount_Balance_Average
clients_without_cobranding_without_package['SavingAccount_Balance_Average'] = np.where(
    clients_without_cobranding_without_package.SavingAccount_Balance_Average.isnull(),
    (clients_without_cobranding_without_package.SavingAccount_Balance_FirstDate + clients_without_cobranding_without_package.SavingAccount_Balance_LastDate) / 2,
    clients_without_cobranding_without_package.SavingAccount_Balance_Average
)

# Transforms features

# Age group to ordinal
age_group_ordinals = {
    'Menor a 18 años': 1800,
    'Entre 18 y 29 años': 1829,
    'Entre 30 y 39 años': 3039,
    'Entre 40 y 49 años': 4049,
    'Entre 50 y 59 años': 5059,
    'Entre 60 y 64 años': 6064,
    'Entre 65 y 69 años': 6569,
    'Mayor a 70 años': 7000,
}
identity_features['Client_Age_grp_ordinal'] = identity_features['Client_Age_grp'].map(age_group_ordinals)

# Sum Insurances
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

# Sum Products
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

# Sum Operations
operation_fields = [column for column in clients_without_cobranding_without_package.columns if 'Operations' in column]
clients_without_cobranding_without_package['Operations'] = clients_without_cobranding_without_package[operation_fields].apply(lambda row: sum(row), axis=1)
for operation_field in operation_fields:
    clients_without_cobranding_without_package[f'Operations_{operation_field}_vs_total'] = np.where(
        clients_without_cobranding_without_package['Operations'] == 0,
        np.floor(clients_without_cobranding_without_package['Operations']),
        np.floor(clients_without_cobranding_without_package[operation_field] / clients_without_cobranding_without_package['Operations'])
    )

# Aggregate features

# Months between products
identity_features['First_product_dt'] = pd.to_datetime(identity_features['First_product_dt'])
identity_features['Last_product_dt'] = pd.to_datetime(identity_features['Last_product_dt'])
identity_features['Month'] = pd.to_datetime(identity_features['Month'])
identity_features['Last_first_product_Months'] = ((identity_features['Last_product_dt'] - identity_features['First_product_dt']).dt.days) / 30
identity_features['Month_first_product_Months'] = ((identity_features['Month'] - identity_features['First_product_dt']).dt.days) / 30
identity_features['Month_last_product_Months'] = ((identity_features['Month'] - identity_features['Last_product_dt']).dt.days) / 30

# Card debit vs credit
clients_without_cobranding_without_package['Card_Debit_vs_credit'] = np.where(
    clients_without_cobranding_without_package['SavingAccount_DebitCard_Spend_Amount'] == 0,
    clients_without_cobranding_without_package['CreditCard_Total_Spending'],
    (clients_without_cobranding_without_package['CreditCard_Total_Spending'] / clients_without_cobranding_without_package['SavingAccount_DebitCard_Spend_Amount'])
)

# SavingAccount debit vs credit
clients_without_cobranding_without_package['SavingAccount_Debit_vs_credit'] = np.where(
    clients_without_cobranding_without_package['SavingAccount_Debits_Amounts'] == 0,
    clients_without_cobranding_without_package['SavingAccount_Credits_Amounts'],
    (clients_without_cobranding_without_package['SavingAccount_Credits_Amounts'] / clients_without_cobranding_without_package['SavingAccount_Debits_Amounts'])
)

# Parse and add data for money fields
# - Add value in CER - Coeficiente de Estabilización de Referencia (Argentina only)
# - Add min, max, mean, median, etc stats values
# - Remove outliers (from min to 3-sigma)
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

def process_money_field(data, field, min=0, sigmas=3):
    new_data = data.copy()

    # Remove data from min
    new_data = new_data[new_data[field] >= min]

    # Remove outliers N sigmas
    sigma = sigmas * new_data[data[field] > min][field].std()
    new_data[field] = np.where(new_data[field] > sigma, sigma, new_data[field])

    # Add CER (Coeficiente de Estabilización de Referencia) value
    field_cer = f'{field}_CER'
    new_data[field_cer] = new_data.apply(calculate_with_cer, args=(field,), axis=1)

    # Add stats columns
    aggregations = new_data.groupby(['client_id'])[[field, field_cer]].agg([
        np.sum, np.amax, np.min, np.mean, np.median, np.count_nonzero, np.var
    ]).reset_index()
    aggregations.columns = [f'{col[0]}_{col[1]}' if col[1] != '' else col[0] for col in aggregations.columns]
    aggregations.reset_index()
    new_data = new_data.merge(
        aggregations[aggregations.columns],
        how='inner',
        on='client_id',
    )

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
clients_with_aggregations = clients_without_cobranding_without_package.copy()
for money_field in money_fields:
    clients_with_aggregations = process_money_field(clients_with_aggregations, money_field, min=0 if money_field != 'SavingAccount_Balance_Average' else 150)

# Print operations vs columns
[print(col) for col in clients_with_aggregations.columns if 'Operations' in col and 'vs' in col]

# Print ordinal columns
[print(col) for col in clients_with_aggregations.columns if 'ordinal' in col]

# Print Active_Insurance columns
[print(col) for col in clients_with_aggregations.columns if 'Active_Insurance' in col]

# Print money columns
[print(col) for col in clients_with_aggregations.columns if 'CreditCard_Balance_ARG' in col and 'CER' not in col]
[print(col) for col in clients_with_aggregations.columns if 'CreditCard_Balance_ARG' in col and 'CER' in col]

show_data_summary(clients_with_aggregations)
