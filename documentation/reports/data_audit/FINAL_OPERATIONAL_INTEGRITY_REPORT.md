# Final Operational Referential Integrity Report

**Project Title**: Autonomous Multi-Agent Customer Support Resolution System with Human-in-the-Loop Escalation  
**Date**: September 28, 2026  

---

## 1. Operational Entity Relationship Audit

The operational dataset in `04_Operational_Data_V4.xlsx` contains 6 relational entities representing the synthetic operational state of the enterprise support environment across E-commerce, Education, and Small Business domains.

| Entity Table | Total Records | Primary Key | Foreign Key Reference | Referential Integrity Status | Domain Consistency Status |
|:---|:---:|:---|:---|:---:|:---:|
| `Customers` | 1,000 | `Customer_ID` | None | **100% PASS** (1,000 unique IDs) | **100% PASS** (334 E-comm, 333 Edu, 333 SmallBiz) |
| `Orders` | 1,500 | `Order_ID` | `Customer_ID` -> `Customers` | **100% PASS** (1,500 valid FKs) | **100% PASS** (All 1,500 E-commerce) |
| `Transactions` | 1,803 | `Transaction_ID` | `Customer_ID` -> `Customers`<br>`Order_ID` -> `Orders` | **100% PASS** (1,803 valid FKs) | **100% PASS** (Tx Customer matches Order Customer exactly) |
| `Support_Tickets`| 1,000 | `Ticket_ID` | `Customer_ID` -> `Customers` | **100% PASS** (1,000 valid FKs) | **100% PASS** (Ticket domain matches Customer domain) |
| `Students` | 500 | `Student_ID` | `Customer_ID` -> `Customers` | **100% PASS** (500 valid FKs) | **100% PASS** (All 500 Education) |
| `Business_Records`| 500 | `Business_Record_ID` | `Customer_ID` -> `Customers` | **100% PASS** (500 valid FKs) | **100% PASS** (All 500 Small Business) |
| `Memory` | 1,500 | `Memory_ID` | `Customer_ID` -> `Customers`<br>`Related_Ticket_ID` -> `Support_Tickets` | **100% PASS** (1,500 valid FKs) | **100% PASS** |

---

## 2. Integrity Constraint Verification Details

1. **Order Customer Domain Constraint**:
   - Evaluated: `Orders.Customer_ID.Domain == 'E-commerce'`
   - Result: 1,500 / 1,500 (100.0% match). 0 cross-domain mismatches.
2. **Transaction Customer vs. Order Customer Constraint**:
   - Evaluated: `Transactions.Customer_ID == Orders[Transaction.Order_ID].Customer_ID`
   - Result: 1,803 / 1,803 (100.0% match). 0 customer discrepancies between order and transaction.
3. **Student Domain Constraint**:
   - Evaluated: `Students.Customer_ID.Domain == 'Education'`
   - Result: 500 / 500 (100.0% match). 0 mismatches.
4. **Business Record Domain Constraint**:
   - Evaluated: `Business_Records.Customer_ID.Domain == 'Small Business'`
   - Result: 500 / 500 (100.0% match). 0 mismatches.
5. **Support Ticket Domain Constraint**:
   - Evaluated: `Support_Tickets.Domain == Customers[Support_Tickets.Customer_ID].Domain`
   - Result: 1,000 / 1,000 (100.0% match). 0 mismatches.
6. **Memory Referential Integrity**:
   - Evaluated: `Memory.Customer_ID in Customers` and `Memory.Related_Ticket_ID in Support_Tickets`
   - Result: 1,500 / 1,500 (100.0% match). 0 orphans.

---

## 3. Numeric & Chronological Range Audits

- **Orders Amount**: 299 to 7,999 INR (Mean: 2,752.4 INR, Median: 2,499.0 INR). No negative amounts.
- **Transactions Amount**: 499 to 50,000 INR (Mean: 6,481.2 INR). No negative amounts.
- **Student Attendance**: 62.0% to 94.0% (Mean: 79.4%). All within valid percentage bounds [0, 100].
- **Memory Chronology**: `Created_Date <= Last_Updated` is TRUE across 100% of rows (Range: 2026-07-15 to 2026-09-01).
- **Missing Value Handling**:
  - `Students.Scholarship_Status`: 120 missing values imputed canonically as `'None'`. All other operational columns have 0 missing values.
