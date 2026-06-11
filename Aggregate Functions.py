import pandas as pd
import re
df1 = pd.read_excel("F:/Loans Analysis/Branch Reports/EMBU BRANCH NEW MEMBERS WITHOUT LOANS.xlsx")
df2 = pd.read_excel("F:/Loans Analysis/Branch Reports/Account Code.xlsx")
print(df1.columns)
print(df2.columns)
def clean_IDNumber(df, col="PhoneNo"):
    
   """ Standardize IDNumber column:
    - Convert to string
    - Remove .0 from Excel numbers
    - Strip leading/trailing spaces
    - Remove non-breaking spaces """

   df[col] = (
        df[col]
        .astype(str)                     # force string
        .str.replace(".0", "", regex=False)  # remove .0 from Excel
        .str.strip()                     # strip spaces
        .str.replace("\u00A0", "", regex=True) # remove non-breaking space
    )
   
   return df
df2 = clean_IDNumber(df2, "PhoneNo")
df1 = clean_IDNumber(df1, "PhoneNo")
df2_subset = df2[["PhoneNo", "Name" , "IDNum"]]

df2_subset_first = (
    df2_subset
    .drop_duplicates(subset="PhoneNo", keep="first")
)
merged_df = df1.merge(
    df2_subset_first,
    on="PhoneNo",
    how="left"
)
with pd.ExcelWriter("F:/Loans Analysis/Branch Reports/MSME Accounts.xlsx",engine = "xlsxwriter")as writer:
    merged_df.to_excel(writer, sheet_name = "summary",index=False)