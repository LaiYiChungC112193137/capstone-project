import pandas as pd

# File path and encoding
path = "/content/國際商港貨物裝卸量.csv"
encoding = "big5"

# Read CSV
df = pd.read_csv(path, encoding=encoding)

# Ensure expected columns exist
expected_cols = {"年月", "港口別", "總計"}
missing = expected_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# Normalize strings (optional but helps avoid whitespace issues)
df["港口別"] = df["港口別"].astype(str).str.strip()
df["年月"] = df["年月"].astype(str).str.strip()

# Convert 年月 (YYYYMM) to datetime (use first day of month) and set as index
df["年月_dt"] = pd.to_datetime(df["年月"], format="%Y%m", errors="coerce")
if df["年月_dt"].isna().any():
    # show problematic rows for debugging
    bad = df[df["年月_dt"].isna()][["年月"]].drop_duplicates()
    raise ValueError(f"Some 年月 values could not be parsed. Examples:\n{bad.head().to_string(index=False)}")

df = df.set_index("年月_dt").sort_index()
df.index.name = "年月"

# Filter 港口別 == "高雄港"
kaohsiung = df[df["港口別"] == "高雄港"].copy()

# Clean 總計 column to numeric (remove commas, parentheses, etc.)
# Adjust regex if there are other formatting characters
kaohsiung["總計"] = (
    kaohsiung["總計"]
    .astype(str)
    .str.replace(",", "", regex=True)
    .str.replace("(", "-", regex=False)
    .str.replace(")", "", regex=False)
)
kaohsiung["總計"] = pd.to_numeric(kaohsiung["總計"], errors="coerce")

# Optionally drop rows where 總計 could not be converted
kaohsiung = kaohsiung[kaohsiung["總計"].notna()]

# Select only the 總計 column (index is 年月)
result = kaohsiung[["總計"]]

# Show result
print(result.head(20))
