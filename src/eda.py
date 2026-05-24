import os
import matplotlib.pyplot as plt
import seaborn as sns

PLOT_DIR = 'outputs/plots'

os.makedirs(PLOT_DIR, exist_ok=True)
sns.set_style('whitegrid')

# CHURN DISTRIBUTION
def churn_distribution(df):
    plt.figure(figsize=(6, 5))
    sns.countplot(x='Churn',data=df)
    plt.title('Customer Churn Distribution')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.savefig(f'{PLOT_DIR}/churn_distribution.png',bbox_inches='tight')
    plt.close()

# GENDER VS CHURN
def gender_vs_churn(df):
    plt.figure(figsize=(6, 5))
    sns.countplot(x='gender',hue='Churn',data=df)
    plt.title('Gender vs Churn')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.savefig(f'{PLOT_DIR}/gender_vs_churn.png',bbox_inches='tight')
    plt.close()

# SENIOR CITIZEN VS CHURN
def seniorcitizen_vs_churn(df):
    plt.figure(figsize=(6, 5))
    sns.countplot(x='SeniorCitizen',hue='Churn',data=df)
    plt.title('Senior Citizen vs Churn')
    plt.xlabel('Senior Citizen')
    plt.ylabel('Count')
    plt.savefig(f'{PLOT_DIR}/seniorcitizen_vs_churn.png',bbox_inches='tight')
    plt.close()

# PARTNER VS CHURN
def partner_vs_churn(df):
    plt.figure(figsize=(6, 5))
    sns.countplot(x='Partner',hue='Churn',data=df)
    plt.title('Partner vs Churn')
    plt.xlabel('Partner')
    plt.ylabel('Count')
    plt.savefig(f'{PLOT_DIR}/partner_vs_churn.png',bbox_inches='tight')
    plt.close()

# DEPENDENTS VS CHURN
def dependents_vs_churn(df):
    plt.figure(figsize=(6, 5))
    sns.countplot(x='Dependents',hue='Churn',data=df)
    plt.title('Dependents vs Churn')
    plt.xlabel('Dependents')
    plt.ylabel('Count')
    plt.savefig(f'{PLOT_DIR}/dependents_vs_churn.png',bbox_inches='tight')
    plt.close()

# CONTRACT VS CHURN
def contract_vs_churn(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Contract',hue='Churn',data=df)
    plt.title('Contract Type vs Churn')
    plt.xlabel('Contract Type')
    plt.ylabel('Count')
    plt.xticks(rotation=10)
    plt.savefig(f'{PLOT_DIR}/contract_vs_churn.png',bbox_inches='tight')
    plt.close()

# INTERNET SERVICE VS CHURN
def internetservice_vs_churn(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(x='InternetService',hue='Churn',data=df)
    plt.title('Internet Service vs Churn')
    plt.xlabel('Internet Service')
    plt.ylabel('Count')
    plt.savefig(f'{PLOT_DIR}/internetservice_vs_churn.png',bbox_inches='tight')
    plt.close()

# PAYMENT METHOD VS CHURN
def paymentmethod_vs_churn(df):
    plt.figure(figsize=(10, 5))
    sns.countplot(x='PaymentMethod',hue='Churn',data=df)
    plt.title('Payment Method vs Churn')
    plt.xlabel('Payment Method')
    plt.ylabel('Count')
    plt.xticks(rotation=20)
    plt.savefig(f'{PLOT_DIR}/paymentmethod_vs_churn.png',bbox_inches='tight')
    plt.close()

# TECH SUPPORT VS CHURN
def techsupport_vs_churn(df):
    plt.figure(figsize=(7, 5))
    sns.countplot(x='TechSupport',hue='Churn',data=df)
    plt.title('Tech Support vs Churn')
    plt.xlabel('Tech Support')
    plt.ylabel('Count')
    plt.savefig(f'{PLOT_DIR}/techsupport_vs_churn.png',bbox_inches='tight')
    plt.close()

# ONLINE SECURITY VS CHURN
def onlinesecurity_vs_churn(df):
    plt.figure(figsize=(7, 5))
    sns.countplot(x='OnlineSecurity',hue='Churn',data=df)
    plt.title('Online Security vs Churn')
    plt.xlabel('Online Security')
    plt.ylabel('Count')
    plt.savefig(f'{PLOT_DIR}/onlinesecurity_vs_churn.png',bbox_inches='tight')
    plt.close()

# TENURE VS CHURN
def tenure_vs_churn(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Churn',y='tenure',data=df)
    plt.title('Tenure vs Churn')
    plt.xlabel('Churn')
    plt.ylabel('Tenure')
    plt.savefig(f'{PLOT_DIR}/tenure_vs_churn.png',bbox_inches='tight')
    plt.close()

# MONTHLY CHARGES VS CHURN
def monthlycharges_vs_churn(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Churn',y='MonthlyCharges',data=df)
    plt.title('Monthly Charges vs Churn')
    plt.xlabel('Churn')
    plt.ylabel('Monthly Charges')
    plt.savefig(f'{PLOT_DIR}/monthlycharges_vs_churn.png',bbox_inches='tight')
    plt.close()

# TOTAL CHARGES DISTRIBUTION
def totalcharges_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df['TotalCharges'],bins=30,kde=True)
    plt.title('Total Charges Distribution')
    plt.xlabel('Total Charges')
    plt.ylabel('Frequency')
    plt.savefig(f'{PLOT_DIR}/totalcharges_distribution.png',bbox_inches='tight')
    plt.close()

# TOTAL SERVICES VS CHURN
def totalservices_vs_churn(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Churn',y='TotalServices',data=df)
    plt.title('Total Services vs Churn')
    plt.xlabel('Churn')
    plt.ylabel('Total Services')
    plt.savefig(f'{PLOT_DIR}/totalservices_vs_churn.png',bbox_inches='tight')
    plt.close()

# CORRELATION HEATMAP
def correlation_heatmap(df):
    plt.figure(figsize=(18, 12))
    correlation = df.corr(numeric_only=True)
    sns.heatmap(correlation,cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig(f'{PLOT_DIR}/correlation_heatmap.png',bbox_inches='tight')
    plt.close()
    
# MONTHLY CONTRACT RISK VS CHURN
def monthlycontractrisk_vs_churn(df):
    plt.figure(figsize=(6, 5))
    sns.countplot(x='MonthlyContractRisk', hue='Churn', data=df)
    plt.title('Monthly Contract Risk vs Churn')
    plt.xlabel('Monthly Contract Risk')
    plt.ylabel('Count')
    plt.savefig(f'{PLOT_DIR}/monthlycontractrisk_vs_churn.png', bbox_inches='tight')
    plt.close()
    

# HIGH VALUE CUSTOMER VS CHURN
def highvaluecustomer_vs_churn(df):
    plt.figure(figsize=(6, 5))
    sns.countplot(x='HighValueCustomer', hue='Churn', data=df)
    plt.title('High Value Customer vs Churn')
    plt.xlabel('High Value Customer')
    plt.ylabel('Count')
    plt.savefig(f'{PLOT_DIR}/highvaluecustomer_vs_churn.png', bbox_inches='tight')
    plt.close()
    
# FEATURE CORRELATION WITH CHURN
def churn_correlation(df):
    plt.figure(figsize=(8, 10))
    correlation = df.corr(numeric_only=True)['Churn'].sort_values(ascending=False)
    sns.barplot(x=correlation.values, y=correlation.index)
    plt.title('Feature Correlation with Churn')
    plt.xlabel('Correlation')
    plt.ylabel('Features')
    plt.savefig(f'{PLOT_DIR}/churn_correlation.png', bbox_inches='tight')
    plt.close()
    
# COMPLETE EDA
def run_eda(df):
    print("\nGenerating EDA Visualizations")
    churn_distribution(df)
    gender_vs_churn(df)
    seniorcitizen_vs_churn(df)
    partner_vs_churn(df)
    dependents_vs_churn(df)
    contract_vs_churn(df)
    internetservice_vs_churn(df)
    paymentmethod_vs_churn(df)
    techsupport_vs_churn(df)
    onlinesecurity_vs_churn(df)
    tenure_vs_churn(df)
    monthlycharges_vs_churn(df)
    totalcharges_distribution(df)
    totalservices_vs_churn(df)
    correlation_heatmap(df)
    monthlycontractrisk_vs_churn(df)
    highvaluecustomer_vs_churn(df)
    churn_correlation(df)
    print("\nEDA Completed Successfully")
    print(f"\nPlots Saved Successfully In: {PLOT_DIR}")