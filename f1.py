# SECTION 1: Importing Libraries and Setting Up Environment
#importing libraries
import pandas as pd   # for data manipulation
import numpy as np  # for numerical operations
import matplotlib.pyplot as plt   # for creating visualizations
import seaborn as sns    # for beautiful statistical graphics
from datetime import datetime  # for date/time operations
import warnings   # to ignore warnings
warnings.filterwarnings('ignore')  # ignoring warnings


# Configure visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

print("All libraries imported successfully!")


# SECTION 2: LOADING ALL DATASETS
print("\n" + "="*80)
print("SECTION 2: LOADING DATA")
print("="*80)

print("\nLoading all CSV files...")

try: 
    # Load all datasets
    circuits = pd.read_csv('csv/circuits.csv')
    constructor_results = pd.read_csv('csv/constructor_results.csv')
    constructor_standings = pd.read_csv('csv/constructor_standings.csv')    
    constructors = pd.read_csv('csv/constructors.csv')
    driver_standings = pd.read_csv('csv/driver_standings.csv')  
    drivers = pd.read_csv('csv/drivers.csv')
    lap_times = pd.read_csv('csv/lap_times.csv')
    pit_stops = pd.read_csv('csv/pit_stops.csv')
    qualifying = pd.read_csv('csv/qualifying.csv')
    races = pd.read_csv('csv/races.csv')
    seasons = pd.read_csv('csv/seasons.csv')
    sprint_results = pd.read_csv('csv/sprint_results.csv')
    status = pd.read_csv('csv/status.csv')

    # Try to load results.csv, if it exists
    try:
        results = pd.read_csv('csv/results.csv')
        has_results = True
        print(f"Loaded results.csv - {len(results):,} race results")
    except FileNotFoundError:
        print("results.csv not found - download it from Kaggle for complete analysis")
        has_results = False

    print(f"Loaded circuits.csv - {len(circuits):,} circuits")
    print(f"Loaded constructor_results.csv - {len(constructor_results):,} constructor results")
    print(f"Loaded constructor_standings.csv - {len(constructor_standings):,} constructor standings")
    print(f"Loaded constructors.csv - {len(constructors):,} constructors")  
    print(f"Loaded driver_standings.csv - {len(driver_standings):,} driver standings")
    print(f"Loaded drivers.csv - {len(drivers):,} drivers")     
    print(f"Loaded lap_times.csv - {len(lap_times):,} lap times")
    print(f"Loaded pit_stops.csv - {len(pit_stops):,} pit stops")
    print(f"Loaded qualifying.csv - {len(qualifying):,} qualifying results")    

except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please ensure all CSV files are in the 'csv' directory and try again.")
    print("Download from: https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2024")
    exit()


# SECTION 3: DATA CLEANING AND PREPROCESSING
print("\n" + "="*80)
print("SECTION 3: DATA CLEANING AND PREPROCESSING")
print("="*80)

print("\n--- Step 1: Inspecting data quality ---")

#function to check data quality
def check_data_quality(df,name):
    """Check and report data quality issues"""
    print(f"\n{name}:")
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Duplicates rows: {df.duplicated().sum()}")
    return df.isnull().sum().sum(), df.duplicated().sum()

# Check quality of main datasets
total_missing = 0
total_duplicates = 0

for df, name in [(drivers, 'driver'), (races, 'races'), (circuits, 'circuits'), (constructors, 'constructors'), (driver_standings, 'driver_standings')]:
    missing, dupes = check_data_quality(df, name)
    total_missing += missing
    total_duplicates += dupes

print(f"\n OVERALL DATA QUALITY:")
print(f"  Total missing values across key datasets: {total_missing}")
print(f"  Total duplicate rows: {total_duplicates}")

print("\n--- Step 2: Handling Missing Values ---")
# Check missing values in drivers
print("\nMissing values in drivers dataset:")
print(drivers.isnull().sum()[drivers.isnull().sum() > 0])

# The '\\N' string is used for NULL in this dataset - replace it
print("\nReplacing '\\N' strings with actual NaN...")
for df in [drivers, races, circuits, constructors, driver_standings, constructor_standings, qualifying, pit_stops, lap_times]: 
    #Replace \\N with NaN in all string columns
    df.replace('\\N', np.nan, inplace=True)

print("Replaced \\N with NaN")

#Check drivers dataset specifically
print("\nDrivers dataset - handling missing values:")
print(f" Missing driver numbers: {drivers['number'].isnull().sum()} (OK - early F1 didn't use numbers)")
print(f" Missing driver codes: {drivers['code'].isnull().sum()} (OK - codes introduced later)")
print(f" Missing URLs: {drivers['url'].isnull().sum()}")

# No critical missing values in drivers - codes and numbers are optional

# Check races dataset
print("\nRaces dataset - handling missing values:")
race_missing = races.isnull().sum()
print(race_missing[race_missing > 0])

# Practice session times are often missing for older races - this is OK
# Sprint dates/times missing for non-sprint races - also OK

# -------------------------------------------------------------------
print("\n--- Step 3: Data Type Conversions ---")
# -------------------------------------------------------------------

print("\nConverting date columns to datetime format...")

# Convert date columns in drivers
drivers['dob'] = pd.to_datetime(drivers['dob'], errors='coerce')
print(f"✓ Converted drivers.dob to datetime")

# Convert date columns in races
races['date'] = pd.to_datetime(races['date'], errors='coerce')
print(f"✓ Converted races.date to datetime")

# Convert time columns (they might be time objects or strings)
# For this analysis, we'll keep them as strings since we mainly need dates

# -------------------------------------------------------------------
print("\n--- Step 4: Creating Derived Features ---")
# -------------------------------------------------------------------

print("\nCreating useful derived columns...")

# 1. Driver age
current_year = 2020  # Dataset goes up to 2020
drivers['age_in_2020'] = current_year - drivers['dob'].dt.year
print(f"✓ Created 'age_in_2020' column in drivers")

# 2. Driver full name
drivers['full_name'] = drivers['forename'] + ' ' + drivers['surname']
print(f"✓ Created 'full_name' column in drivers")

# 3. Constructor full info (for merging)
constructors['full_info'] = constructors['name'] + ' (' + constructors['nationality'] + ')'
print(f"✓ Created 'full_info' column in constructors")

# 4. Decade column in races (for grouping by era)
races['decade'] = (races['year'] // 10) * 10
print(f"✓ Created 'decade' column in races")

# 5. Convert lap time milliseconds to seconds
if len(lap_times) > 0:
    lap_times['lap_seconds'] = lap_times['milliseconds'] / 1000
    print(f"✓ Created 'lap_seconds' column in lap_times")

# 6. Convert pit stop milliseconds to seconds
if len(pit_stops) > 0:
    pit_stops['duration_seconds'] = pit_stops['milliseconds'] / 1000
    print(f"✓ Created 'duration_seconds' column in pit_stops")

# -------------------------------------------------------------------
print("\n--- Step 5: Handling Duplicates ---")
# -------------------------------------------------------------------

print("\nChecking for and removing duplicates...")

initial_len = len(driver_standings)
driver_standings.drop_duplicates(inplace=True)
removed = initial_len - len(driver_standings)
print(f"  driver_standings: Removed {removed} duplicates")

initial_len = len(constructor_standings)
constructor_standings.drop_duplicates(inplace=True)
removed = initial_len - len(constructor_standings)
print(f"  constructor_standings: Removed {removed} duplicates")

# -------------------------------------------------------------------
print("\n--- Step 6: Data Validation ---")
# -------------------------------------------------------------------

print("\nValidating data ranges and consistency...")

# Check for invalid years
min_year = races['year'].min()
max_year = races['year'].max()
print(f"  Year range: {min_year} to {max_year} ✓")

# Check for negative points (shouldn't exist)
negative_points = driver_standings[driver_standings['points'] < 0]
if len(negative_points) > 0:
    print(f"  ⚠ Found {len(negative_points)} records with negative points - removing")
    driver_standings = driver_standings[driver_standings['points'] >= 0]
else:
    print(f"  No negative points found ✓")

# Check for invalid positions
invalid_positions = driver_standings[driver_standings['position'] < 1]
if len(invalid_positions) > 0:
    print(f"  ⚠ Found {len(invalid_positions)} records with invalid positions - removing")
    driver_standings = driver_standings[driver_standings['position'] >= 1]
else:
    print(f"  No invalid positions found ✓")

# -------------------------------------------------------------------
print("\n--- Step 7: Outlier Detection ---")
# -------------------------------------------------------------------

print("\nDetecting outliers in key metrics...")

# Check pit stop durations
if len(pit_stops) > 0:
    # Reasonable pit stop range: 1 second to 60 seconds
    outlier_pit_stops = pit_stops[
        (pit_stops['duration_seconds'] < 1) | 
        (pit_stops['duration_seconds'] > 60)
    ]
    print(f"  Pit stops outside normal range (1-60s): {len(outlier_pit_stops)}")
    print(f"    These might be penalties, errors, or unusual circumstances - keeping for analysis")

# Check lap times
if len(lap_times) > 0:
    # Sample to avoid memory issues
    lap_sample = lap_times.sample(min(100000, len(lap_times)))
    # Reasonable lap time: 60 seconds to 180 seconds (1-3 minutes)
    outlier_laps = lap_sample[
        (lap_sample['lap_seconds'] < 60) | 
        (lap_sample['lap_seconds'] > 180)
    ]
    print(f"  Lap times outside normal range (60-180s): {len(outlier_laps)} in sample")
    print(f"    These might be safety car laps, pit laps, or data errors")

print("\n✅ DATA CLEANING COMPLETE!")
print(f"\nCleaned datasets ready for analysis:")
print(f"  • {len(drivers)} drivers")
print(f"  • {len(races)} races")
print(f"  • {len(circuits)} circuits")
print(f"  • {len(constructors)} constructors")
print(f"  • {len(driver_standings):,} driver standings records")
print(f"  • {len(constructor_standings):,} constructor standings records")

# ==============================================================================
# SECTION 4: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
print("\n" + "="*80)
print("SECTION 4: EXPLORATORY DATA ANALYSIS")
print("="*80)

print("\n--- Basic Statistics ---")

# Drivers statistics
print("\nDRIVERS:")
print(f"  Total drivers: {len(drivers)}")
print(f"  Nationalities: {drivers['nationality'].nunique()}")
print(f"  Age range: {drivers['age_in_2020'].min():.0f} to {drivers['age_in_2020'].max():.0f} years")
print(f"  Average age (in 2020): {drivers['age_in_2020'].mean():.1f} years")

# Races statistics
print("\nRACES:")
print(f"  Total races: {len(races)}")
print(f"  Years covered: {races['year'].min()} to {races['year'].max()}")
print(f"  Total circuits used: {races['circuitId'].nunique()}")
print(f"  Countries hosted races: {circuits['country'].nunique()}")

# Championship statistics
print("\nCHAMPIONSHIPS:")
championship_years = driver_standings[driver_standings['position'] == 1]['raceId'].nunique()
print(f"  Championship seasons: {championship_years}")

# Constructor statistics
print("\nCONSTRUCTORS:")
print(f"  Total constructors: {len(constructors)}")
print(f"  Constructor nationalities: {constructors['nationality'].nunique()}")

print("\n--- Top Statistics ---")

# Most common nationalities
top_nationalities = drivers['nationality'].value_counts().head(5)
print("\nTop 5 Driver Nationalities:")
for nat, count in top_nationalities.items():
    print(f"  {nat}: {count} drivers")

# Most raced circuits
circuit_races = races.groupby('circuitId').size().sort_values(ascending=False)
circuit_info = circuits.set_index('circuitId')
print("\nTop 5 Most Raced Circuits:")
for circuit_id, count in circuit_races.head(5).items():
    circuit_name = circuit_info.loc[circuit_id, 'name']
    print(f"  {circuit_name}: {count} races")

# ==============================================================================
# SECTION 5: ANALYSIS & VISUALIZATIONS
# ==============================================================================
print("\n" + "="*80)
print("SECTION 5: DETAILED ANALYSIS & VISUALIZATIONS")
print("="*80)

# Create output directory for plots
import os
if not os.path.exists('f1_visualizations'):
    os.makedirs('f1_visualizations')
    print("✓ Created 'f1_visualizations' folder for outputs")

# ============================================================================
# ANALYSIS 1: F1 EVOLUTION OVER TIME
# ============================================================================
print("\n" + "-"*80)
print("ANALYSIS 1: How has Formula 1 evolved over the decades?")
print("-"*80)

# Number of races per season
races_per_season = races.groupby('year').size().reset_index(name='num_races')

# Unique circuits per decade
circuits_per_decade = races.groupby('decade')['circuitId'].nunique().reset_index()
circuits_per_decade.columns = ['decade', 'num_circuits']

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Races per season
axes[0].plot(races_per_season['year'], races_per_season['num_races'], 
             linewidth=2.5, color='#E10600', marker='o', markersize=4,
             markerfacecolor='white', markeredgecolor='#E10600', markeredgewidth=2)
axes[0].set_xlabel('Year', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Number of Races', fontweight='bold', fontsize=12)
axes[0].set_title('F1 Calendar Expansion: Races per Season (1950-2020)', 
                  fontweight='bold', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Add annotations for key points
max_races_year = races_per_season.loc[races_per_season['num_races'].idxmax()]
axes[0].annotate(f'Peak: {max_races_year["num_races"]:.0f} races',
                xy=(max_races_year['year'], max_races_year['num_races']),
                xytext=(max_races_year['year']-10, max_races_year['num_races']+2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, fontweight='bold', color='red')

# Plot 2: Unique circuits per decade
bars = axes[1].bar(circuits_per_decade['decade'], circuits_per_decade['num_circuits'], 
                   width=8, color='#1e88e5', edgecolor='black', linewidth=1.5)
axes[1].set_xlabel('Decade', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Number of Unique Circuits', fontweight='bold', fontsize=12)
axes[1].set_title('Circuit Diversity: Unique Venues per Decade', 
                  fontweight='bold', fontsize=14)
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('f1_visualizations/01_evolution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_evolution.png")
plt.close()

# Key insights
print(f"\n💡 KEY INSIGHTS:")
print(f"   • F1 started with {races_per_season['num_races'].iloc[0]} races in 1950")
print(f"   • Peak season: {max_races_year['num_races']:.0f} races in {max_races_year['year']:.0f}")
print(f"   • Modern seasons average {races_per_season['num_races'].tail(10).mean():.1f} races")
print(f"   • Most diverse decade: {circuits_per_decade.loc[circuits_per_decade['num_circuits'].idxmax(), 'decade']:.0f}s")

# ============================================================================
# ANALYSIS 2: CHAMPIONSHIP WINNERS
# ============================================================================
print("\n" + "-"*80)
print("ANALYSIS 2: Who are the greatest F1 World Champions?")
print("-"*80)

# Get championship winners (position = 1 in final standings)
championships = driver_standings[driver_standings['position'] == 1].copy()
championships = championships.merge(races[['raceId', 'year']], on='raceId')
championships = championships.merge(drivers[['driverId', 'full_name', 'nationality']], on='driverId')

# Count championships per driver
championship_counts = championships.groupby('full_name').agg({
    'year': 'count',
    'nationality': 'first'
}).reset_index()
championship_counts.columns = ['driver_name', 'championships', 'nationality']
championship_counts = championship_counts.sort_values('championships', ascending=False)

# Total points by driver
total_points = driver_standings.groupby('driverId')['points'].sum().reset_index()
total_points = total_points.merge(drivers[['driverId', 'full_name', 'nationality']], on='driverId')
total_points = total_points.sort_values('points', ascending=False).head(15)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Championships
top_15_champs = championship_counts.head(15)
colors_champ = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(top_15_champs)))

axes[0].barh(range(len(top_15_champs)), top_15_champs['championships'], 
             color=colors_champ, edgecolor='black', linewidth=1.5)
axes[0].set_yticks(range(len(top_15_champs)))
axes[0].set_yticklabels(top_15_champs['driver_name'])
axes[0].set_xlabel('World Championships Won', fontweight='bold', fontsize=12)
axes[0].set_title('F1 Legends: Most World Championships', fontweight='bold', fontsize=14)
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

# Add value labels
for i, v in enumerate(top_15_champs['championships']):
    axes[0].text(v + 0.1, i, str(v), va='center', fontweight='bold', fontsize=10)

# Plot 2: Career points
colors_points = plt.cm.viridis(np.linspace(0, 1, len(total_points)))
axes[1].barh(range(len(total_points)), total_points['points'], 
             color=colors_points, edgecolor='black', linewidth=1)
axes[1].set_yticks(range(len(total_points)))
axes[1].set_yticklabels(total_points['full_name'])
axes[1].set_xlabel('Career Points', fontweight='bold', fontsize=12)
axes[1].set_title('Highest Career Points (All Time)', fontweight='bold', fontsize=14)
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('f1_visualizations/02_champions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_champions.png")
plt.close()

# Print top 5
print(f"\n🏆 TOP 5 WORLD CHAMPIONS:")
for i, row in championship_counts.head(5).iterrows():
    print(f"   {row['driver_name']}: {row['championships']} championships ({row['nationality']})")

# ============================================================================
# ANALYSIS 3: CONSTRUCTOR DOMINANCE
# ============================================================================
print("\n" + "-"*80)
print("ANALYSIS 3: Which teams have dominated Formula 1?")
print("-"*80)

# Constructor championships
constructor_champs = constructor_standings[constructor_standings['position'] == 1].copy()
constructor_champs = constructor_champs.merge(races[['raceId', 'year']], on='raceId')
constructor_champs = constructor_champs.merge(constructors[['constructorId', 'name', 'nationality']], 
                                              on='constructorId')

# Count titles
constructor_titles = constructor_champs.groupby('name').agg({
    'year': 'count',
    'nationality': 'first'
}).reset_index()
constructor_titles.columns = ['constructor', 'titles', 'nationality']
constructor_titles = constructor_titles.sort_values('titles', ascending=False).head(12)

# Total points
constructor_points = constructor_standings.groupby('constructorId')['points'].sum().reset_index()
constructor_points = constructor_points.merge(
    constructors[['constructorId', 'name', 'nationality']], on='constructorId'
)
constructor_points = constructor_points.sort_values('points', ascending=False).head(12)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Championships
team_colors = []
for name in constructor_titles['constructor']:
    if 'Ferrari' in name:
        team_colors.append('#E10600')
    elif 'Mercedes' in name:
        team_colors.append('#00D2BE')
    elif 'Williams' in name:
        team_colors.append('#0600EF')
    elif 'McLaren' in name:
        team_colors.append('#FF8700')
    elif 'Red Bull' in name:
        team_colors.append('#0600EF')
    else:
        team_colors.append('#666666')

axes[0].barh(range(len(constructor_titles)), constructor_titles['titles'], 
             color=team_colors, edgecolor='black', linewidth=1.5)
axes[0].set_yticks(range(len(constructor_titles)))
axes[0].set_yticklabels(constructor_titles['constructor'])
axes[0].set_xlabel('Constructor Championships', fontweight='bold', fontsize=12)
axes[0].set_title('Constructor Dominance: Most Championships', fontweight='bold', fontsize=14)
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

# Add value labels
for i, v in enumerate(constructor_titles['titles']):
    axes[0].text(v + 0.2, i, str(v), va='center', fontweight='bold', fontsize=10)

# Plot 2: Total points
colors_con = plt.cm.plasma(np.linspace(0, 1, len(constructor_points)))
axes[1].barh(range(len(constructor_points)), constructor_points['points'], 
             color=colors_con, edgecolor='black', linewidth=1)
axes[1].set_yticks(range(len(constructor_points)))
axes[1].set_yticklabels(constructor_points['name'])
axes[1].set_xlabel('Career Points', fontweight='bold', fontsize=12)
axes[1].set_title('Constructor Career Points', fontweight='bold', fontsize=14)
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('f1_visualizations/03_constructors.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_constructors.png")
plt.close()

print(f"\n🏁 TOP 5 CONSTRUCTORS:")
for i, row in constructor_titles.head(5).iterrows():
    print(f"   {row['constructor']}: {row['titles']} championships ({row['nationality']})")

# ============================================================================
# ANALYSIS 4: CIRCUIT ANALYSIS
# ============================================================================
print("\n" + "-"*80)
print("ANALYSIS 4: Most iconic F1 circuits")
print("-"*80)

# Most raced circuits
circuit_count = races.groupby('circuitId').size().reset_index(name='num_races')
circuit_count = circuit_count.merge(circuits[['circuitId', 'name', 'country', 'location']], 
                                    on='circuitId')
circuit_count = circuit_count.sort_values('num_races', ascending=False).head(15)

# Countries with most circuits
country_circuits = circuits.groupby('country').size().reset_index(name='num_circuits')
country_circuits = country_circuits.sort_values('num_circuits', ascending=False).head(12)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Most raced circuits
colors_circuit = plt.cm.Greens(np.linspace(0.4, 0.9, len(circuit_count)))
axes[0].barh(range(len(circuit_count)), circuit_count['num_races'], 
             color=colors_circuit, edgecolor='darkgreen', linewidth=1.5)
axes[0].set_yticks(range(len(circuit_count)))
axes[0].set_yticklabels([f"{row['name'][:30]}" for _, row in circuit_count.iterrows()])
axes[0].set_xlabel('Number of Races Held', fontweight='bold', fontsize=12)
axes[0].set_title('Most Iconic F1 Circuits (by races held)', fontweight='bold', fontsize=14)
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

# Plot 2: Countries
colors_country = plt.cm.Oranges(np.linspace(0.4, 0.9, len(country_circuits)))
bars = axes[1].bar(range(len(country_circuits)), country_circuits['num_circuits'], 
                   color=colors_country, edgecolor='black', linewidth=1.5)
axes[1].set_xticks(range(len(country_circuits)))
axes[1].set_xticklabels(country_circuits['country'], rotation=45, ha='right')
axes[1].set_ylabel('Number of Circuits', fontweight='bold', fontsize=12)
axes[1].set_title('Countries with Most F1 Circuits', fontweight='bold', fontsize=14)
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('f1_visualizations/04_circuits.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_circuits.png")
plt.close()

print(f"\n🏎️ TOP 5 CIRCUITS:")
for i, row in circuit_count.head(5).iterrows():
    print(f"   {row['name']} ({row['country']}): {row['num_races']} races")

# ============================================================================
# ANALYSIS 5: PIT STOP EVOLUTION
# ============================================================================
print("\n" + "-"*80)
print("ANALYSIS 5: Evolution of pit stop strategy")
print("-"*80)

# Merge pit stops with races to get years
pit_analysis = pit_stops.merge(races[['raceId', 'year']], on='raceId')

# Average pit stop duration by year
pit_yearly = pit_analysis.groupby('year')['duration_seconds'].agg([
    ('avg', 'mean'),
    ('median', 'median'),
    ('fastest', 'min'),
    ('count', 'count')
]).reset_index()

# Pit stops per race over time
stops_per_race = pit_analysis.groupby(['raceId', 'year']).size().reset_index(name='num_stops')
avg_stops_yearly = stops_per_race.groupby('year')['num_stops'].mean().reset_index()

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Duration trends
axes[0].plot(pit_yearly['year'], pit_yearly['avg'], 
             label='Average', linewidth=2.5, marker='o', color='#D32F2F', markersize=5)
axes[0].plot(pit_yearly['year'], pit_yearly['median'], 
             label='Median', linewidth=2.5, marker='s', color='#1976D2', markersize=5)
axes[0].plot(pit_yearly['year'], pit_yearly['fastest'], 
             label='Fastest', linewidth=2.5, marker='^', color='#388E3C', markersize=5)
axes[0].set_xlabel('Year', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Duration (seconds)', fontweight='bold', fontsize=12)
axes[0].set_title('Pit Stop Speed Evolution: Getting Faster Every Year', 
                  fontweight='bold', fontsize=14)
axes[0].legend(fontsize=11, loc='upper right')
axes[0].grid(True, alpha=0.3)

# Highlight the improvement
if len(pit_yearly) > 1:
    first_avg = pit_yearly['avg'].iloc[0]
    last_avg = pit_yearly['avg'].iloc[-1]
    improvement = ((first_avg - last_avg) / first_avg) * 100
    axes[0].text(0.02, 0.98, f'Speed improvement: {improvement:.1f}%', 
                transform=axes[0].transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=11, fontweight='bold')

# Plot 2: Number of stops
axes[1].bar(avg_stops_yearly['year'], avg_stops_yearly['num_stops'], 
            color='#7B1FA2', edgecolor='black', linewidth=0.8, alpha=0.8)
axes[1].set_xlabel('Year', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Average Stops per Race', fontweight='bold', fontsize=12)
axes[1].set_title('Pit Stop Strategy: Average Stops per Race', 
                  fontweight='bold', fontsize=14)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('f1_visualizations/05_pitstops.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_pitstops.png")
plt.close()

# Find fastest pit stop
fastest_stop = pit_analysis.loc[pit_analysis['duration_seconds'].idxmin()]
print(f"\n⚡ FASTEST PIT STOP EVER: {fastest_stop['duration_seconds']:.3f} seconds")
print(f"   Average in recent years: {pit_yearly['avg'].tail(5).mean():.2f} seconds")

# ============================================================================
# ANALYSIS 6: DRIVER NATIONALITIES
# ============================================================================
print("\n" + "-"*80)
print("ANALYSIS 6: Global reach of Formula 1")
print("-"*80)

# Drivers by nationality
driver_nations = drivers.groupby('nationality').size().reset_index(name='count')
driver_nations = driver_nations.sort_values('count', ascending=False).head(15)

# Constructors by nationality
constructor_nations = constructors.groupby('nationality').size().reset_index(name='count')
constructor_nations = constructor_nations.sort_values('count', ascending=False).head(12)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Drivers
colors_drv = plt.cm.Blues(np.linspace(0.4, 0.9, len(driver_nations)))
axes[0].barh(range(len(driver_nations)), driver_nations['count'], 
             color=colors_drv, edgecolor='black', linewidth=1.5)
axes[0].set_yticks(range(len(driver_nations)))
axes[0].set_yticklabels(driver_nations['nationality'])
axes[0].set_xlabel('Number of Drivers', fontweight='bold', fontsize=12)
axes[0].set_title('F1 Drivers by Nationality (Top 15)', fontweight='bold', fontsize=14)
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

# Add percentages
total_drivers = len(drivers)
for i, (_, row) in enumerate(driver_nations.iterrows()):
    pct = (row['count'] / total_drivers) * 100
    axes[0].text(row['count'] + 1, i, f"{row['count']} ({pct:.1f}%)", 
                va='center', fontsize=9)

# Plot 2: Constructors
colors_con = plt.cm.Reds(np.linspace(0.4, 0.9, len(constructor_nations)))
axes[1].barh(range(len(constructor_nations)), constructor_nations['count'], 
             color=colors_con, edgecolor='black', linewidth=1.5)
axes[1].set_yticks(range(len(constructor_nations)))
axes[1].set_yticklabels(constructor_nations['nationality'])
axes[1].set_xlabel('Number of Constructors', fontweight='bold', fontsize=12)
axes[1].set_title('F1 Constructors by Nationality (Top 12)', fontweight='bold', fontsize=14)
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('f1_visualizations/06_nationality.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_nationality.png")
plt.close()

print(f"\n🌍 TOP 5 NATIONS (Drivers):")
for i, row in driver_nations.head(5).iterrows():
    pct = (row['count'] / total_drivers) * 100
    print(f"   {row['nationality']}: {row['count']} drivers ({pct:.1f}%)")

# ============================================================================
# ANALYSIS 7: LAP TIME TRENDS (Sampled)
# ============================================================================
print("\n" + "-"*80)
print("ANALYSIS 7: Lap time evolution (F1 getting faster)")
print("-"*80)

# Sample lap times (to avoid memory issues)
print("Sampling lap times for analysis (using 200,000 random laps)...")
lap_sample = lap_times.sample(min(200000, len(lap_times)), random_state=42)
lap_sample = lap_sample.merge(races[['raceId', 'year']], on='raceId')

# Average lap time by year
lap_yearly = lap_sample.groupby('year')['lap_seconds'].mean().reset_index()

# Visualization
plt.figure(figsize=(14, 7))

plt.plot(lap_yearly['year'], lap_yearly['lap_seconds'], 
         linewidth=3, marker='o', markersize=6, color='#0D47A1',
         markerfacecolor='white', markeredgecolor='#0D47A1', markeredgewidth=2)

plt.xlabel('Year', fontweight='bold', fontsize=12)
plt.ylabel('Average Lap Time (seconds)', fontweight='bold', fontsize=12)
plt.title('F1 Speed Evolution: Average Lap Times Over the Years', 
          fontweight='bold', fontsize=14)
plt.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(lap_yearly['year'], lap_yearly['lap_seconds'], 1)
p = np.poly1d(z)
plt.plot(lap_yearly['year'], p(lap_yearly['year']), 
         "r--", alpha=0.8, linewidth=2, label='Trend')

plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig('f1_visualizations/07_lap_times.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 07_lap_times.png")
plt.close()

if len(lap_yearly) > 1:
    first_avg = lap_yearly['lap_seconds'].iloc[0]
    last_avg = lap_yearly['lap_seconds'].iloc[-1]
    improvement = ((first_avg - last_avg) / first_avg) * 100
    print(f"\n🚀 LAP TIME IMPROVEMENT: {improvement:.1f}% faster than early years")

# ============================================================================
# ANALYSIS 8: CHAMPIONSHIP BATTLES (How close were they?)
# ============================================================================
print("\n" + "-"*80)
print("ANALYSIS 8: Championship battle intensity")
print("-"*80)

# Get final standings of each season
final_standings = driver_standings.merge(races[['raceId', 'year', 'round']], on='raceId')

# Get last race of each season
last_races = races.groupby('year')['round'].max().reset_index()
last_races.columns = ['year', 'final_round']

# Filter to final standings only
final_only = final_standings.merge(last_races, on='year')
final_only = final_only[final_only['round'] == final_only['final_round']]

# Get champion and runner-up
champions = final_only[final_only['position'] == 1][['year', 'points', 'driverId']].copy()
champions.columns = ['year', 'champion_points', 'champion_id']

runners_up = final_only[final_only['position'] == 2][['year', 'points', 'driverId']].copy()
runners_up.columns = ['year', 'runner_points', 'runner_id']

# Merge
battles = champions.merge(runners_up, on='year')
battles['margin'] = battles['champion_points'] - battles['runner_points']

# Add champion names
battles = battles.merge(drivers[['driverId', 'full_name']], 
                       left_on='champion_id', right_on='driverId')
battles = battles.rename(columns={'full_name': 'champion'})

# Visualization
plt.figure(figsize=(14, 7))

# Color by closeness
colors = ['darkred' if m < 10 else 'orange' if m < 25 else 'green' 
          for m in battles['margin']]

plt.bar(battles['year'], battles['margin'], color=colors, 
        edgecolor='black', linewidth=0.8, alpha=0.7)

# Add reference lines
plt.axhline(y=10, color='red', linestyle='--', linewidth=2, alpha=0.6, 
           label='Very Close (<10 pts)')
plt.axhline(y=25, color='orange', linestyle='--', linewidth=2, alpha=0.6, 
           label='Close (<25 pts)')

plt.xlabel('Year', fontweight='bold', fontsize=12)
plt.ylabel('Points Margin (Champion - Runner-up)', fontweight='bold', fontsize=12)
plt.title('Championship Battle Intensity: How Close Were The Fights?', 
          fontweight='bold', fontsize=14)
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('f1_visualizations/08_championship_margins.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 08_championship_margins.png")
plt.close()

# Stats
closest = battles.nsmallest(5, 'margin')
print(f"\n⚔️ CLOSEST CHAMPIONSHIPS:")
for _, row in closest.iterrows():
    print(f"   {row['year']:.0f}: {row['champion']} won by {row['margin']:.0f} points")

# ==============================================================================
# SECTION 6: SUMMARY & KEY FINDINGS
# ==============================================================================
print("\n" + "="*80)
print("SECTION 6: SUMMARY & KEY FINDINGS")
print("="*80)

print(f"""
📊 COMPREHENSIVE F1 DATA ANALYSIS SUMMARY
{'='*80}

DATA CLEANED & ANALYZED:
  ✓ {len(drivers)} drivers from {drivers['nationality'].nunique()} countries
  ✓ {len(races)} races across {races['year'].max() - races['year'].min() + 1} years
  ✓ {len(circuits)} unique circuits in {circuits['country'].nunique()} countries
  ✓ {len(constructors)} constructors
  ✓ {len(pit_stops):,} pit stops analyzed
  ✓ {len(lap_times):,} lap times recorded

KEY FINDINGS:

1. 🏎️ F1 EVOLUTION
   • Calendar grew from {races_per_season['num_races'].iloc[0]} races (1950) to {races_per_season['num_races'].iloc[-1]} races (2020)
   • Peak season: {max_races_year['num_races']:.0f} races in {max_races_year['year']:.0f}
   • F1 has visited {circuits['circuitId'].nunique()} unique circuits worldwide

2. 🏆 CHAMPIONSHIP LEGENDS
   • Top champion: {championship_counts.iloc[0]['driver_name']} ({championship_counts.iloc[0]['championships']} titles)
   • {len(championship_counts[championship_counts['championships'] >= 3])} drivers won 3+ championships
   • {len(championship_counts)} different drivers have been world champion

3. 🏁 TEAM DOMINANCE
   • Most successful constructor: {constructor_titles.iloc[0]['constructor']} ({constructor_titles.iloc[0]['titles']} titles)
   • British teams have won {constructor_titles[constructor_titles['nationality'] == 'British']['titles'].sum()} championships
   • Italian heritage (Ferrari) has {constructor_titles[constructor_titles['constructor'] == 'Ferrari']['titles'].sum()} titles

4. 🗺️ GLOBAL REACH
   • {driver_nations.iloc[0]['nationality']} has produced most drivers ({driver_nations.iloc[0]['count']})
   • European dominance: {len(driver_nations[driver_nations['nationality'].isin(['British', 'French', 'German', 'Italian'])])} of top 15 nations
   • Sport becoming more global with drivers from {drivers['nationality'].nunique()} nationalities

5. ⚡ TECHNOLOGY EVOLUTION
   • Pit stops improved by {improvement:.1f}% (duration)
   • Fastest pit stop: {fastest_stop['duration_seconds']:.3f} seconds
   • Modern teams can change 4 tires in under 2 seconds!

6. ⚔️ CHAMPIONSHIP BATTLES
   • Average championship margin: {battles['margin'].mean():.1f} points
   • Closest championship: {battles['margin'].min():.0f} points
   • {len(battles[battles['margin'] < 10])} championships decided by less than 10 points

ALL VISUALIZATIONS SAVED IN: f1_visualizations/
  ✓ 01_evolution.png - F1 growth over time
  ✓ 02_champions.png - Greatest drivers
  ✓ 03_constructors.png - Team dominance
  ✓ 04_circuits.png - Iconic venues
  ✓ 05_pitstops.png - Pit stop evolution
  ✓ 06_nationality.png - Global reach
  ✓ 07_lap_times.png - Speed improvements
  ✓ 08_championship_margins.png - Close battles

NEXT STEPS FOR YOUR PROJECT:
  1. Add your own analyses (use this as a template!)
  2. Create interactive dashboards with Plotly/Dash
  3. Build predictive models (who will win next race?)
  4. Analyze specific eras (V10 vs V8 vs Hybrid)
  5. Deep dive into specific drivers/teams
  6. Present findings in a blog post or presentation

""")

print("="*80)
print("🎉 ANALYSIS COMPLETE! Check 'f1_visualizations' folder for all charts.")
print("="*80)
      


