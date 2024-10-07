import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
st.set_page_config(
    page_title="Stock Quantity Analysis",
    layout="wide",  # Use "wide" layout for more screen space
    initial_sidebar_state="expanded"  # Expand the sidebar by default
)

st.title("Sales Analysis from December 2021 to December 2023")
st.markdown('---')
st.header ('Demand Forecasting System for Optimizing Inventory and Supply Chain Efficiency Using Historical Sales Data ')
st.markdown('---')

st.subheader("Summary of Product data :")
st.markdown("""
            * There are lot of duplicates for each StockCode.
            * Most of the duplicates are distored in sentence.
            * Example of duplicates for a sample "Small Popcorn Opener"  are as "Popcorn Opener Small" , "Small Opener PopCorn"
            """)
st.subheader("Summary of Customers Data :")
st.markdown(""" * Most of the customers are not in the our Customer database.
                * All customers in database are from United Kingdom, France, USA, Belgium, Australia, Netherlands.
            """)

st.subheader("Summary of Transactions data :")
st.markdown("""
            * There are lot of transactions missing the Customer ID.
            * There are negative values in Quantity , assuming that they are returned by customers or products brought by company.
            * There are negative values for Price which is assumed that money is paid by company i.e outflow.
            * The Revenue is calculated as the product of the Quantity and Price. i.e, Revenue = Quantity * Price
            * We assumed a new Customer ID for all null values.
            """)

transactions_01 = pd.read_csv('Transactional_data_retail_01.csv')
transactions_02 = pd.read_csv('Transactional_data_retail_02.csv')

customer_data = pd.read_csv('CustomerDemographics.csv')
product_info = pd.read_csv('Productinfo.csv')
customer_data.loc[customer_data.shape[0]]= {'Customer ID':'00000','Country':'Unknown'}
values = customer_data.Country.value_counts().values.tolist()
countries = customer_data.Country.value_counts().keys().tolist()
col1,col2 = st.columns(2)
with col1:
    plt.figure(figsize=(8,6))
    plt.pie(x=values,labels=countries,autopct='%1.1f%%')
    plt.title('Distributions of Customer by Countries.')
    st.pyplot(plt)

# removing the duplicates
product_info['Description'] = product_info['Description'].str.strip()
product_info.drop_duplicates(inplace=True)

transactions_01.drop_duplicates(inplace=True)
transactions_02.drop_duplicates(inplace=True)
# Filling the nan values with the specified rule used before
transactions_01['Customer ID'].fillna("00000",inplace=True)
transactions_02['Customer ID'].fillna("00000",inplace=True)


# Calculating the top 10 StockCodes based on the Quantity
st.markdown("""
                ---
                The Top 10 Stocks by quantity is are the following.
            ---
""")
top_10_quantity_01_sales = transactions_01.groupby('StockCode').Quantity.sum().nlargest(10)
top_10_stockcodes = list(top_10_quantity_01_sales.to_dict().keys())
top_10_quantity = list(top_10_quantity_01_sales.to_dict().values())
col1, col2 = st.columns(2)
top_10_quantity_02_sales = transactions_02.groupby('StockCode').Quantity.sum().nlargest(10)
top_10_stockcodes_02 = list(top_10_quantity_02_sales.to_dict().keys())
top_10_quantity_02 = list(top_10_quantity_02_sales.to_dict().values())

# First chart in the first column
with col1:
    plt.figure(figsize=(12, 9))
    plt.bar(x=top_10_stockcodes, height=top_10_quantity, color='skyblue', edgecolor='black')
    plt.xlabel('Stock Codes', fontsize=14)
    plt.ylabel('Quantity', fontsize=14)
    plt.title('Top 10 Stock Codes by Quantity (Dataset 01)', fontsize=16, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i in range(len(top_10_quantity)):
        plt.text(i, top_10_quantity[i] + 0.5, str(top_10_quantity[i]), ha='center', fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)

# Second chart in the second column
with col2:
    plt.figure(figsize=(12, 9))
    plt.bar(x=top_10_stockcodes_02, height=top_10_quantity_02, color='skyblue', edgecolor='black')
    plt.xlabel('Stock Codes', fontsize=14)
    plt.ylabel('Quantity', fontsize=14)
    plt.title('Top 10 Stock Codes by Quantity (Dataset 02)', fontsize=16, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i in range(len(top_10_quantity_02)):
        plt.text(i, top_10_quantity_02[i] + 0.5, str(top_10_quantity_02[i]), ha='center', fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)


# calculating the revenue 
transactions_01['Revenue'] = transactions_01['Price']*transactions_01['Quantity']
transactions_02['Revenue'] = transactions_02['Price']*transactions_02['Quantity']
st.markdown("""
                ---
                The Top 10 Stocks by Revenue is are the following.
            ---
""")
top_10_revenue_01_sales = transactions_01.groupby('StockCode').Revenue.sum().nlargest(10)
top_10_stockcodes = list(top_10_revenue_01_sales.to_dict().keys())
top_10_revenue = list(top_10_revenue_01_sales.to_dict().values())
top_10_revenue_02_sales = transactions_02.groupby('StockCode').Revenue.sum().nlargest(10)
top_10_stockcodes_02 = list(top_10_revenue_02_sales.to_dict().keys())
top_10_revenue_02 = list(top_10_revenue_02_sales.to_dict().values())
col1, col2 = st.columns(2)
# First chart in the first column
with col1:
    plt.figure(figsize=(12, 9))
    plt.bar(x=top_10_stockcodes, height=top_10_revenue, color='skyblue', edgecolor='black')
    plt.xlabel('Stock Codes', fontsize=14)
    plt.ylabel('Revenue', fontsize=14)
    plt.title('Top 10 Stock Codes by Revenue (Dataset 01)', fontsize=16, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i in range(len(top_10_revenue)):
        plt.text(i, top_10_revenue[i] + 0.5, str(top_10_revenue[i]), ha='center', fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)

# Second chart in the second column
with col2:
    plt.figure(figsize=(12, 9))
    plt.bar(x=top_10_stockcodes_02, height=top_10_revenue_02, color='skyblue', edgecolor='black')
    plt.xlabel('Stock Codes', fontsize=14)
    plt.ylabel('Revenue', fontsize=14)
    plt.title('Top 10 Stock Codes by Revenue (Dataset 02)', fontsize=16, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i in range(len(top_10_revenue_02)):
        plt.text(i, top_10_revenue_02[i] + 0.5, str(top_10_revenue_02[i]), ha='center', fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)

# combining data entirely
merged_data = pd.concat([transactions_01,transactions_02],axis=0)
merged_top_10 = merged_data.groupby('StockCode').Quantity.sum().nlargest(10)
merged_top_10_stockcodes = list(merged_top_10.keys())
merged_top_10_quantity = list(merged_top_10.values)
st.markdown("""
                ---
                The Top 10 Stocks by Quantity from 2021 December to 2023 December
            ---
""")
plt.figure(figsize=(8, 4))
plt.bar(x=merged_top_10_stockcodes, height=merged_top_10_quantity, color='skyblue', edgecolor='black')
plt.xlabel('Stock Codes', fontsize=14)
plt.ylabel('Quantity', fontsize=14)
plt.title('Top 10 Stock Codes by Quantity from 2021-2023', fontsize=16, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i in range(len(merged_top_10_quantity)):
    plt.text(i, merged_top_10_quantity[i] + 0.5, str(merged_top_10_quantity[i]), ha='center', fontsize=12)
plt.tight_layout()
st.pyplot(plt)

# Creating a demand forecast for the each stock
models = {}
transactions_01['date'] = pd.to_datetime(transactions_01['InvoiceDate'],dayfirst=True)
transactions_02['date'] = pd.to_datetime(transactions_02['InvoiceDate'],dayfirst=True)
transactions_01['date'] = transactions_01['date'].dt.date
transactions_02['date'] = transactions_02['date'].dt.date
select_stock = st.selectbox('Select a stock to see the forecast',merged_top_10_stockcodes)
st.write(select_stock)
merged_data = pd.concat([transactions_01,transactions_02],axis=0)
selected_data = merged_data[merged_data.StockCode==select_stock]
required_data = selected_data.loc[:,['Quantity','date']]
required_data['date'] = pd.to_datetime(required_data['date'])
required_data.set_index('date',inplace=True)
weekly_data = required_data.resample("W").sum()
train_size = int(len(weekly_data) * 0.8)
train_data = weekly_data.loc[:weekly_data.index[train_size],:]
test_data = weekly_data.loc[weekly_data.index[train_size]:,:]
model = ARIMA(train_data['Quantity'],order=(1,1,1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test_data))
mse = mean_squared_error(test_data['Quantity'], forecast)
rmse = np.sqrt(mse)
st.write(f'Root Mean Squared Error: {rmse}')
forecast_index = pd.date_range(start=train_data.index[-1] + pd.Timedelta(weeks=1), periods=15, freq='W')
st.write(forecast)
forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecast'])
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['Quantity'], label='Historical Data', marker='o')
plt.plot(test_data.index, test_data['Quantity'], label='Test Data', marker='o',color='orange')
plt.plot(test_data.index,forecast[:len(test_data)], label='Forecast', marker='o', color='green')
plt.title('Weekly Data and 15-Week Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()
st.pyplot(plt)

