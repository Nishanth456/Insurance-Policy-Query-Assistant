import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

# Initialize Faker for fake names
fake = Faker()

# Define policy types with coverage and premium ranges
policy_types = {
    "Health": {"coverage_range": (100000, 500000), "premium_range": (300, 1000)},
    "Auto": {"coverage_range": (50000, 300000), "premium_range": (200, 800)},
    "Life": {"coverage_range": (200000, 1000000), "premium_range": (500, 2000)},
    "Home": {"coverage_range": (100000, 700000), "premium_range": (400, 1500)},
    "Travel": {"coverage_range": (25000, 150000), "premium_range": (100, 600)},
}

# Generate 100 rows of fake data
data = []
for i in range(1, 101):
    policy_type = random.choice(list(policy_types.keys()))
    coverage = random.randint(*policy_types[policy_type]["coverage_range"])
    premium = random.randint(*policy_types[policy_type]["premium_range"])
    renewal_date = datetime.now() + timedelta(days=random.randint(30, 365))

    # Round values
    rounded_coverage = round(coverage, -3)  # Round to nearest 1000
    rounded_premium = round(premium, -2)    # Round to nearest 100

    data.append({
        "policy_id": f"POL{str(i).zfill(3)}",
        "customer_name": fake.name(),
        "policy_type": policy_type,
        "coverage_amount": rounded_coverage,
        "premium": rounded_premium,
        "renewal_date": renewal_date.strftime("%Y-%m-%d")
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("insurance_policies_sample_100_final.csv", index=False)
print("Dataset saved as insurance_policies_sample_100_final.csv")
