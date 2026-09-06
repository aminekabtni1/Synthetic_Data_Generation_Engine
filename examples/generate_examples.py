"""Generate demo datasets: ecommerce and healthcare."""
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)

# E-commerce
n = 2000
customers = pd.DataFrame({
    "customer_id": np.arange(1, 501),
    "age": rng.normal(35, 10, 500).astype(int).clip(18, 70),
    "country": rng.choice(["USA", "UK", "Germany", "France", "Canada"], 500, p=[0.4, 0.15, 0.15, 0.15, 0.15]),
    "signup_date": pd.to_datetime(rng.integers(pd.Timestamp("2020-01-01").value // 10**9, pd.Timestamp("2023-01-01").value // 10**9, 500), unit="s"),
    "is_premium": rng.choice([True, False], 500, p=[0.3, 0.7]),
})
# Orders with FK
orders = []
for i in range(1, n+1):
    cust = rng.choice(customers["customer_id"])
    # Correlated: premium customers spend more
    is_prem = customers.loc[customers["customer_id"]==cust, "is_premium"].values[0]
    amount = rng.normal(120 if is_prem else 60, 30) + rng.normal(0, 10)
    amount = max(5, amount)
    qty = max(1, int(rng.normal(3 if is_prem else 2, 1)))
    category = rng.choice(["Electronics", "Clothing", "Books", "Home"], p=[0.3,0.3,0.2,0.2])
    # City correlated with country via customer
    country = customers.loc[customers["customer_id"]==cust, "country"].values[0]
    city_map = {"USA": ["NYC","LA","Chicago"], "UK": ["London","Manchester"], "Germany": ["Berlin","Munich"], "France": ["Paris","Lyon"], "Canada": ["Toronto","Vancouver"]}
    city = rng.choice(city_map[country])
    orders.append([i, cust, round(amount,2), qty, category, city, country, pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(rng.integers(0,365)))])
orders_df = pd.DataFrame(orders, columns=["order_id","customer_id","amount","quantity","category","city","country","order_date"])
customers.to_csv("examples/customers.csv", index=False)
orders_df.to_csv("examples/ecommerce.csv", index=False)
print(f"Ecommerce: customers {customers.shape}, orders {orders_df.shape}")

# Healthcare (synthetic sensitive-feeling)
n_pat = 1000
health = pd.DataFrame({
    "patient_id": [f"P{i:05d}" for i in range(1, n_pat+1)],
    "age": rng.normal(50, 15, n_pat).astype(int).clip(20, 85),
    "gender": rng.choice(["M","F"], n_pat),
    "bmi": rng.normal(27, 4, n_pat).round(1),
    "blood_pressure": (rng.normal(120, 15, n_pat) + 0.3*(rng.normal(27,4,n_pat)-27)*2).astype(int).clip(90,200),
    "cholesterol": (rng.normal(200, 30, n_pat) + 0.4*(rng.normal(50,15,n_pat)-50)).astype(int).clip(120,350),
    "smoker": rng.choice(["Yes","No"], n_pat, p=[0.25,0.75]),
    "diagnosis": rng.choice(["Healthy","Hypertension","Diabetes","Cardio"], n_pat, p=[0.5,0.2,0.15,0.15]),
    "visit_date": pd.to_datetime(rng.integers(pd.Timestamp("2022-01-01").value // 10**9, pd.Timestamp("2024-01-01").value // 10**9, n_pat), unit="s"),
})
# Correlated: smoker more likely cardio
# Adjust diagnosis for smokers
for idx in health[health["smoker"]=="Yes"].sample(frac=0.3, random_state=42).index:
    health.loc[idx, "diagnosis"] = rng.choice(["Cardio","Hypertension"], p=[0.6,0.4])
health.to_csv("examples/healthcare.csv", index=False)
print(f"Healthcare: {health.shape}")
print(health.head())
