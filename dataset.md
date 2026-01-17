# Dataset Description

## Overview
This dataset contains banking customer information for predicting transaction behavior. The dataset includes demographic data, account information, and transaction metrics.

## Target Variable
- **`Avg_Trans_Amount`**: Average transaction amount of the customer (in VND). This is the target variable for prediction.

## Feature Columns

### Customer Information
| Column | Description | Data Type |
|--------|-------------|-----------|
| `Customer_number` | Unique identifier for each customer | Integer |
| `Client_gender` | Gender of the customer (0: Female, 1: Male) | Binary |
| `Age` | Age of the customer (in years) | Float |
| `Staff` | Whether the customer is a bank staff member (0: No, 1: Yes) | Binary |
| `Tenure` | Number of years the customer has been with the bank | Float |

### Service Registration
| Column | Description | Data Type |
|--------|-------------|-----------|
| `SMS` | SMS banking registration status (0: Not registered, 1: Registered) | Binary |
| `Verify_method` | Verification method used (0, 1, 2 representing different methods) | Categorical |
| `EB_register_channel` | E-banking registration channel (0, 1, 2 representing different channels) | Categorical |

### Activity Metrics
| Column | Description | Data Type |
|--------|-------------|-----------|
| `No_Activity_Name` | Number of different activity types performed by the customer | Integer |
| `Type_Transactions` | Number of different transaction types | Integer |
| `Avg_Trans_no_month` | Average number of transactions per month | Float |

### Current Account Information
| Column | Description | Data Type |
|--------|-------------|-----------|
| `No_CurrentAccount` | Number of current accounts owned by the customer | Integer |
| `Avg_CurrentAccount_Balance` | Average balance in current accounts (VND) | Float |
| `Max_CurrentAccount_Balance` | Maximum balance in current accounts (VND) | Float |
| `Min_CurrentAccount_Balance` | Minimum balance in current accounts (VND) | Float |

### Term Deposit Information
| Column | Description | Data Type |
|--------|-------------|-----------|
| `No_TermDeposit` | Number of term deposit accounts | Integer |
| `Avg_TermDeposit_Balance` | Average balance in term deposit accounts (VND) | Float |
| `Max_TermDeposit_Balance` | Maximum balance in term deposit accounts (VND) | Float |
| `Min_TermDeposit_Balance` | Minimum balance in term deposit accounts (VND) | Float |

### Loan Information
| Column | Description | Data Type |
|--------|-------------|-----------|
| `No_Loan` | Number of loan accounts | Integer |
| `Avg_Loan_Balance` | Average loan balance (VND) | Float |
| `Max_Loan_Balance` | Maximum loan balance (VND) | Float |
| `Min_Loan_Balance` | Minimum loan balance (VND) | Float |

### Card Information
| Column | Description | Data Type |
|--------|-------------|-----------|
| `No_CC` | Number of credit cards | Integer |
| `No_DC` | Number of debit cards | Integer |

### Target & Label
| Column | Description | Data Type |
|--------|-------------|-----------|
| `Churn` | Customer churn status (0: Active, 1: Churned) | Binary |

---

## Notes
- All monetary values are in Vietnamese Dong (VND)
- Balance columns can have negative values (e.g., overdraft situations)
- The `Avg_Trans_Amount` (target variable) represents the customer's average transaction value and is used for regression prediction
