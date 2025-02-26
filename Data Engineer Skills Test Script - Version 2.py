"""
Cancer Institute NSW - Data Engineer Skills Test
Author: Dylan Bai
Date: 18th February 2024

This script is designed to complete the assignment provided by the Cancer Institute NSW for the Data Engineer position.
The script ingests synthetic cancer registry data from the UK Simulacrum v2.1.0 dataset, performs data cleaning and analysis,
and generates summaries and visualisations as per the brief requirements.

The script is written in Python and uses the pandas, matplotlib, and seaborn libraries for data manipulation, analysis,
and visualisation.
"""

# ----------------------------
# Section 1: Import Required Modules
# ----------------------------
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For data visualisation
import seaborn as sns  # For enhanced data visualisation

# Set the style for seaborn plots
sns.set(style="whitegrid")

# ----------------------------
# Section 2: Define File Paths
# ----------------------------
# File paths for datasets
patients_file_path = r"C:\Users\dylanbai\3.0 Dyl Personal Stuff\JA\2025 Job Applications\Job Applications\NSW Health - Data Engineer\Homework\simulacrum_v2.1.0\Data\sim_av_patient.csv"
tumours_file_path = r"C:\Users\dylanbai\3.0 Dyl Personal Stuff\JA\2025 Job Applications\Job Applications\NSW Health - Data Engineer\Homework\simulacrum_v2.1.0\Data\sim_av_tumour.csv"
lookup_tables_file_path = r"C:\Users\dylanbai\3.0 Dyl Personal Stuff\JA\2025 Job Applications\Job Applications\NSW Health - Data Engineer\Homework\simulacrum_v2.1.0\Documents\all_z_lookup_tables.xlsx"

# Columns used in the analysis
# Patients dataset columns: 'PATIENTID', 'GENDER', 'ETHNICITY', 'VITALSTATUS', 'VITALSTATUSDATE', 'QUINTILE_2019'
# Tumours dataset columns: 'PATIENTID', 'AGE', 'COMORBIDITIES_27_03', 'BEHAVIOUR_ICD10_O2', 'GRADE', 'STAGE_BEST'

# ----------------------------
# Section 3: Load Lookup Tables
# ----------------------------
# Load and process lookup tables
lookup_tables = pd.read_excel(lookup_tables_file_path, sheet_name=None)

# Create lookup dictionaries
lookup_dicts = {
    'gender': dict(zip(lookup_tables['z_gender']['Code'], lookup_tables['z_gender']['Description'])),
    'ethnicity': dict(zip(lookup_tables['z_ethnicity']['Code'], lookup_tables['z_ethnicity']['Description'])),
    'vitalstatus': dict(zip(lookup_tables['z_vitalstatus']['Code'], lookup_tables['z_vitalstatus']['Description'])),
    'deathlocation': dict(zip(lookup_tables['z_deathlocationcode']['Code'], lookup_tables['z_deathlocationcode']['Description'])),
    'stage': dict(zip(lookup_tables['z_stage']['Code'], lookup_tables['z_stage']['Description'])),
    'comorbidities': dict(zip(lookup_tables['z_comorbidities']['Code'], lookup_tables['z_comorbidities']['Description'])),
    'behaviour': dict(zip(lookup_tables['z_behaviour']['Code'], lookup_tables['z_behaviour']['Description'])),
    'cancercareplanintent': dict(zip(lookup_tables['z_cancercareplanintent']['Code'], lookup_tables['z_cancercareplanintent']['Description'])),
    'grade': dict(zip(lookup_tables['z_grade']['Code'], lookup_tables['z_grade']['Description'])),
    'laterality': dict(zip(lookup_tables['z_laterality']['Code'], lookup_tables['z_laterality']['Description'])),
    'performancestatus': dict(zip(lookup_tables['z_performancestatus']['Code'], lookup_tables['z_performancestatus']['Description']))
}

# ----------------------------
# Section 4: Load and Preview Data
# ----------------------------
# Load datasets
patients = pd.read_csv(patients_file_path)
tumours = pd.read_csv(tumours_file_path)

# Display dataset previews
print("Dataset Previews:")
for name, df in [("Patients", patients), ("Tumours", tumours)]:
    print(f"\n{name} dataset preview:")
    print(df.head())
    print(f"\nShape: {df.shape}")

# ----------------------------
# Section 5: Merge Datasets - Link Patients and Tumours Datasets
# ----------------------------
# Drop the GENDER column from the tumours dataset
# Since both datasets have a GENDER column and they are identical, we only need one.
# We will keep the GENDER column from the patients dataset and drop the one from tumours.
# Finally, merge the datasets on PATIENTID
tumours = tumours.drop(columns=['GENDER'])  # Remove duplicate GENDER column
merged_data = pd.merge(patients, tumours, on='PATIENTID', how='inner')

print("\nMerged dataset information:")
print(f"Shape: {merged_data.shape}")
print("\nColumns:", merged_data.columns.tolist())
print("\nSummary statistics:")
print(merged_data.describe())

# ----------------------------
# Section 6: Apply Lookup Tables
# ----------------------------
# Map codes to descriptions
field_mappings = {
    'GENDER': lookup_dicts['gender'],
    'ETHNICITY': lookup_dicts['ethnicity'],
    'VITALSTATUS': lookup_dicts['vitalstatus'],
    'DEATHLOCATIONCODE': lookup_dicts['deathlocation'],
    'STAGE_BEST': lookup_dicts['stage'],
    'COMORBIDITIES_27_03': lookup_dicts['comorbidities'],
    'BEHAVIOUR_ICD10_O2': lookup_dicts['behaviour'],
    'CANCERCAREPLANINTENT': lookup_dicts['cancercareplanintent'],
    'GRADE': lookup_dicts['grade'],
    'LATERALITY': lookup_dicts['laterality'],
    'PERFORMANCESTATUS': lookup_dicts['performancestatus']
}

for field, mapping in field_mappings.items():
    merged_data[field] = merged_data[field].map(mapping)

# Display the updated dataset
print("\nMerged dataset with decoded categorical variables:")
print(merged_data.head())
print(merged_data.info())

# ----------------------------
# Section 7: Summarise Missing Values
# ----------------------------
# Summarise missing values in the merged dataset
missing_values = merged_data.isnull().sum()
print("\nMissing values summary:")
print(missing_values[missing_values > 0])  # Show only fields with missing values

# Visualise missing values using a bar chart (sorted)
missing_values_sorted = missing_values.sort_values(ascending=False)
plt.figure(figsize=(12, 6))
missing_values_sorted.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Count of Missing Values per Data Field (Sorted)", fontsize=16)
plt.xlabel("Data Fields", fontsize=14)
plt.ylabel("Count of Missing Values", fontsize=14)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add gridlines for better readability
plt.tight_layout()  # Adjust layout to prevent label overlap
plt.show()

# ----------------------------
# Section 8: Data Cleaning
# ----------------------------
# Define essential columns
essential_columns = [
    "PATIENTID", "GENDER", "ETHNICITY", "DEATHLOCATIONCODE", "VITALSTATUS",
    "VITALSTATUSDATE", "TUMOURID", "DIAGNOSISDATEBEST", "BEHAVIOUR_ICD10_O2",
    "STAGE_BEST", "GRADE", "AGE", "LATERALITY", "CANCERCAREPLANINTENT",
    "PERFORMANCESTATUS", "CHRL_TOT_27_03", "COMORBIDITIES_27_03", "QUINTILE_2019"
]

# Clean dataset
merged_data = (merged_data[essential_columns]
              .drop_duplicates(subset=['PATIENTID', 'TUMOURID']))

# Report cleaning results
print(f"\nCleaned dataset shape: {merged_data.shape}")
print(f"Missing values after cleaning:")
print(merged_data.isnull().sum()[merged_data.isnull().sum() > 0])

# ----------------------------
# Section 9: Create a New Column for the Year of VitalStatusDate and show the VITALSTATUSYEAR Breakdown
# ----------------------------
# Process vital status dates
merged_data['VITALSTATUSDATE'] = pd.to_datetime(merged_data['VITALSTATUSDATE'])
merged_data['VITALSTATUSYEAR'] = merged_data['VITALSTATUSDATE'].dt.year

# Analyse vital status year distribution
vital_status_summary = pd.DataFrame({
    "Count": merged_data['VITALSTATUSYEAR'].value_counts().sort_index(),
    "Percentage": (merged_data['VITALSTATUSYEAR'].value_counts().sort_index() / len(merged_data) * 100)
})

print("\nVital Status Year Distribution:")
print(vital_status_summary)

# Count occurrences of each VITALSTATUSYEAR
vital_status_counts = merged_data['VITALSTATUSYEAR'].value_counts().sort_index()

# Create a bar chart for the patient distribution per VITALSTATUSYEAR
plt.figure(figsize=(10, 6))
plt.bar(vital_status_counts.index, vital_status_counts.values, color='skyblue', edgecolor='black')
plt.xlabel("Vital Status Year", fontsize=12)
plt.ylabel("Number of Patients", fontsize=12)
plt.title("Breakdown of Patients by Vital Status Year", fontsize=14)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ----------------------------
# Section 10.0: Summarise Age and Sex Distribution of Patients
# ----------------------------
def analyse_demographics(data, measure):
    """Calculate demographic statistics by gender."""
    # Calculate mode separately since it's not a direct aggregation method
    mode_by_gender = data.groupby('GENDER')[measure].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None
    )
    
    # Calculate other statistics
    stats = data.groupby('GENDER')[measure].agg([
        'mean',
        'median',
        lambda x: x.max() - x.min(),  # range
        lambda x: x.quantile(0.25),   # lower quartile
        lambda x: x.quantile(0.75),   # upper quartile
        'var',
        'std'
    ]).round(2)
    
    # Add mode to the statistics DataFrame
    stats['mode'] = mode_by_gender
    
    # Rename columns for clarity
    stats.columns = ['Mean', 'Median', 'Range', 
                    'Lower Quartile', 'Upper Quartile',
                    'Variance', 'Standard Deviation', 'Mode']
    
    return stats

# Generate age-sex distribution analysis
age_sex_stats = analyse_demographics(merged_data, 'AGE')

# Print the table in a single, structured format
print("\nAge Statistics by Gender:\n")
print(age_sex_stats.to_string())

# Create visualisations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Histogram
sns.histplot(data=merged_data, x='AGE', hue='GENDER', kde=True, 
             bins=30, palette="Set2", ax=ax1)
ax1.set_title("Age Distribution by Gender")
ax1.set_xlabel("Age")
ax1.set_ylabel("Count")

# Box plot
sns.boxplot(data=merged_data, x="GENDER", y="AGE", 
            palette="Set2", ax=ax2)
ax2.set_title("Age Distribution by Gender")
ax2.set_xlabel("Gender")
ax2.set_ylabel("Age")
ax2.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()
    
# ----------------------------
# Section 11: Distribution Analysis and Bar Chart Visualisation for other Data Fields
# ----------------------------
def analyse_distribution(data, column, title):
    """Analyse and visualise distribution of clinical characteristics."""
    # Calculate distribution
    distribution = data[column].value_counts()
    percentage = (distribution / distribution.sum() * 100).round(1)
    
    # Create summary table
    summary = pd.DataFrame({
        'Count': distribution,
        'Percentage (%)': percentage
    })
    
    print(f"\n{title} Distribution:")
    print(summary)
    print(f"Most Common {title}: {distribution.index[0]}")
    
    # Create visualisation
    plt.figure(figsize=(10, 6))
    bars = plt.bar(distribution.index, distribution.values, 
                   color=plt.cm.Paired.colors[:len(distribution)])
    
    # Add percentage labels
    for bar, pct in zip(bars, percentage):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{pct}%', ha='center', va='bottom')
    
    plt.title(f"Distribution of {title}")
    plt.xlabel(title)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Analyse key clinical characteristics
for field, title in [
    ('BEHAVIOUR_ICD10_O2', 'Behaviour'),
    ('GRADE', 'Grade'),
    ('STAGE_BEST', 'Stage')
]:
    analyse_distribution(merged_data, field, title)

# ----------------------------
# Section 12: Summarise Comorbidities of Patients
# ----------------------------
# Clean 'COMORBIDITIES_27_03' column
# Remove unwanted characters, leading zeros, and split multiple values
comorbidities_expanded = (
    tumours[['PATIENTID', 'COMORBIDITIES_27_03']]
    .astype(str)
    .assign(COMORBIDITIES_27_03=lambda df: df['COMORBIDITIES_27_03']
            .str.replace(r"['0]", "", regex=True)
            .str.lstrip('0')
            .str.split(','))
    .explode('COMORBIDITIES_27_03')
    .reset_index(drop=True)
)

# Convert to numeric, drop NaNs, and map to descriptions
comorbidities_expanded['COMORBIDITIES_27_03'] = (
    pd.to_numeric(comorbidities_expanded['COMORBIDITIES_27_03'], errors='coerce', downcast='integer')
    .map(lookup_dicts['comorbidities'])
    .dropna()
)

# Summarise comorbidity counts and percentages
comorbidity_counts = (
    comorbidities_expanded['COMORBIDITIES_27_03']
    .value_counts()
    .reset_index()
    .rename(columns={'COMORBIDITIES_27_03': 'Comorbidities', 'index': 'COMORBIDITY'})
)
comorbidity_counts['PERCENTAGE'] = (comorbidity_counts['count'] / comorbidity_counts['count'].sum()) * 100

# Plot distribution
plt.figure(figsize=(10, 6))
bars = plt.bar(comorbidity_counts['Comorbidities'], comorbidity_counts['count'], color=plt.cm.Paired.colors[:len(comorbidity_counts)])
plt.xlabel('Comorbidity')
plt.ylabel('Count')
plt.title('Distribution of Comorbidities')
plt.xticks(rotation=45, ha='right')
for bar, percentage in zip(bars, comorbidity_counts['PERCENTAGE']):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.show()

# Display results
print("\nComorbidities Summary:")
print(comorbidity_counts)
print(f"Total count of comorbidities: {comorbidity_counts['count'].sum()}")

# ----------------------------
# Section 13: Ethnicity and Income-Level Analysis
# ----------------------------
# Clean income data
merged_data['QUINTILE_2019'] = merged_data['QUINTILE_2019'].str[0].astype(int)

# 13a) Summarise the percentage of each ethnic group for the most deprived income-level group
# Analyse most deprived group
most_deprived = merged_data[merged_data['QUINTILE_2019'] == 1]
ethnicity_deprived = most_deprived['ETHNICITY'].value_counts()
ethnicity_deprived_pct = (ethnicity_deprived / len(most_deprived) * 100).round(1)

# Get top ethnicities and combine others
top_ethnicities = ethnicity_deprived_pct.nlargest(4)
other_pct = ethnicity_deprived_pct[~ethnicity_deprived_pct.index.isin(top_ethnicities.index)].sum()
final_distribution = pd.concat([top_ethnicities, pd.Series({'OTHER': other_pct})])

# Create pie chart
plt.figure(figsize=(8, 8))
plt.pie(final_distribution, labels=final_distribution.index, 
        autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors[:5])
plt.title('Ethnic Distribution in Most Deprived Income Group')
plt.axis('equal')
plt.show()

# 13b) Identify ethnic groups with a higher proportion of lower income earners
# Define lower income earners - I have made the assumption of defining this as QUINTILE_2019 <= 2
lower_income = merged_data[merged_data['QUINTILE_2019'] <= 2]

# Calculate the total ethnic population for each ethnic group
ethnicity_total_population = merged_data.groupby('ETHNICITY').size()

# Calculate the proportion of lower income earners within each ethnic group
ethnicity_lower_income = lower_income.groupby('ETHNICITY').size() / ethnicity_total_population

# Create a summary DataFrame
ethnicity_summary = pd.DataFrame({
    'Proportion of lower income earners': ethnicity_lower_income,
    'Total population count': ethnicity_total_population
})

# Display the original result as a table
print("\nEthnic groups with a higher proportion of lower income earners:")
print(ethnicity_summary.sort_values(by='Proportion of lower income earners', ascending=False))

# -----------------------------
# Additional filtering for population count >= 5000
# -----------------------------
# Filter to include only ethnic groups with a total population count of at least 5,000
ethnicity_summary_filtered = ethnicity_summary[ethnicity_summary['Total population count'] >= 5000]

# Sort by proportion of lower income earners in descending order
ethnicity_summary_sorted = ethnicity_summary_filtered.sort_values(by='Proportion of lower income earners', ascending=False)

# Display the result for filtered groups with population >= 5000
print("\nEthnic groups with a higher proportion of lower income earners (Population count >= 5000):")
print(ethnicity_summary_sorted)

# Step to create the bar chart for Proportion of Lower Income Earners
plt.figure(figsize=(10, 6))
plt.bar(ethnicity_summary_sorted.index, ethnicity_summary_sorted['Proportion of lower income earners'], color='skyblue')
plt.title('Proportion of Lower Income Earners by Ethnicity (Population >= 5000)', fontsize=14)
plt.xlabel('Ethnicity', fontsize=12)
plt.ylabel('Proportion of Lower Income Earners', fontsize=12)
plt.xticks(rotation=45, ha='right')
# Display the percentage on top of each bar
for i, v in enumerate(ethnicity_summary_sorted['Proportion of lower income earners']):
    plt.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()  # Adjust layout to prevent label cut-off
plt.show()

# ----------------------------
# End of Script
# ----------------------------