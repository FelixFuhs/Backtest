import yaml
import pandas as pd
import yfinance as yf
import os
from pathlib import Path
from datetime import datetime, timedelta

# --- Configuration ---
CONFIG_FILE = "datasets.yaml"
CACHE_DIR = Path("cache")
CACHE_EXPIRY_DAYS_YFINANCE = 1  # How old can yfinance cached data be before re-downloading
CACHE_EXPIRY_DAYS_CSV = 30 # Or some other logic for CSVs, e.g., check file modification time

# --- Helper Functions ---

def load_config(config_path=CONFIG_FILE):
    """Loads the dataset configuration from the YAML file."""
    # TODO: Implement YAML loading
    # Ensure to handle file not found errors
    print(f"Loading configuration from {config_path}...")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not config or 'datasets' not in config:
            print("Error: YAML config is empty or 'datasets' key is missing.")
            return None
        return config['datasets']
    except FileNotFoundError:
        print(f"Error: Configuration file {config_path} not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        return None

def get_cache_filepath(dataset_name: str) -> Path:
    """Generates the filepath for a cached dataset."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True) # Ensure cache directory exists
    # We'll use parquet for caching as it's efficient
    return CACHE_DIR / f"{dataset_name.replace(' ', '_').lower()}.parquet"

def load_from_cache(dataset_name: str, expiry_days: int) -> pd.DataFrame | None:
    """
    Loads a dataset from cache if it exists and is not expired.
    For CSV sources, expiry_days might be less relevant than source file modification time.
    """
    cache_file = get_cache_filepath(dataset_name)
    if cache_file.exists():
        file_mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - file_mod_time < timedelta(days=expiry_days):
            print(f"Loading '{dataset_name}' from cache: {cache_file}")
            try:
                return pd.read_parquet(cache_file)
            except Exception as e:
                print(f"Error loading {dataset_name} from cache file {cache_file}: {e}. Will try to refetch.")
                return None
        else:
            print(f"Cache for '{dataset_name}' is expired. Will refetch.")
    return None

def save_to_cache(dataset_name: str, df: pd.DataFrame):
    """Saves a DataFrame to the cache."""
    cache_file = get_cache_filepath(dataset_name)
    print(f"Saving '{dataset_name}' to cache: {cache_file}")
    try:
        df.to_parquet(cache_file)
    except Exception as e:
        print(f"Error saving {dataset_name} to cache file {cache_file}: {e}")

def fetch_from_yfinance(identifier: str, start_date: str | None, end_date: str | None, data_fields: list) -> pd.DataFrame | None:
    """Fetches data from Yahoo Finance using yfinance library."""
    print(f"Fetching '{identifier}' from yfinance (start: {start_date}, end: {end_date})...")
    try:
        # Using auto_adjust=True gives OHLCV already adjusted.
        # 'Close' will be the adjusted close.
        data = yf.download(identifier, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if data.empty:
            print(f"Warning: No data returned from yfinance for {identifier}")
            return None

        # If 'Adj Close' is requested and 'Close' exists (it should, as it's the adjusted price),
        # create 'Adj Close' as a copy of 'Close' for consistency with requested fields.
        if 'Adj Close' in data_fields and 'Close' in data.columns:
            if 'Adj Close' not in data.columns: # Only if yfinance didn't provide it (expected with auto_adjust=True)
                 data['Adj Close'] = data['Close']

        # Select only the requested data_fields
        # Make sure to handle cases where some fields might not be available
        # (though yfinance with auto_adjust=True usually provides OHLCAV)
        available_fields = [field for field in data_fields if field in data.columns]
        missing_fields = [field for field in data_fields if field not in data.columns]
        if missing_fields:
            print(f"Warning: For {identifier}, yfinance did not return the following requested fields: {missing_fields}. 'Close' is adjusted if auto_adjust=True.")

        return data[available_fields]

    except Exception as e:
        print(f"Error fetching data for {identifier} from yfinance: {e}")
        return None

def load_from_csv(file_path_str: str, date_column: str, value_column: str | None = None, data_fields: list | None = None) -> pd.DataFrame | None:
    """Loads data from a CSV file."""
    file_path = Path(file_path_str)
    print(f"Loading from CSV: {file_path} (date_col: {date_column}, value_col: {value_column})")
    try:
        # Read the CSV. We might need to skip rows if there's header text, like in DTB3.csv
        # For DTB3.csv, it seems the actual data starts from the second row if the first is a source note.
        # Let's assume for now a simple CSV structure or that it's clean.
        # If header rows are an issue, pandas read_csv has `skiprows` parameter.
        # For the original DTB3.csv, the values are sometimes '.', which pandas needs to handle as NaN.
        df = pd.read_csv(file_path, parse_dates=[date_column], na_values=['.'])
        df.set_index(date_column, inplace=True)
        df.index.name = 'Date' # Standardize index name

        # If specific data_fields are requested, select them.
        # Otherwise, if only value_column is given, assume it's a series to be named.
        if data_fields:
            # Ensure all requested fields exist
            available_fields = [field for field in data_fields if field in df.columns]
            if not all(field in df.columns for field in data_fields):
                 print(f"Warning: Not all requested data_fields {data_fields} found in {file_path}. Available: {df.columns.tolist()}")
            df = df[available_fields]
        elif value_column and value_column in df.columns:
            # If it's a simple series like the risk-free rate
            df = df[[value_column]]
        else:
            print(f"Warning: CSV {file_path} - value_column '{value_column}' not found or no data_fields specified correctly. Returning all columns.")

        # Convert column names to a standard format if needed (e.g., PascalCase from yfinance)
        # For now, we assume columns from CSV are already as desired or will be handled by user.
        # Example: df.columns = [col.replace(' ', '') for col in df.columns] # Basic cleaning

        return df

    except FileNotFoundError:
        print(f"Error: CSV file {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading data from CSV {file_path}: {e}")
        return None

# --- Main Loading Function ---

def load_dataset(dataset_config: dict) -> pd.DataFrame | None:
    """Loads a single dataset based on its configuration."""
    name = dataset_config.get('name', 'UnknownDataset')
    source_type = dataset_config.get('source_type')
    identifier = dataset_config.get('identifier')
    
    print(f"\nProcessing dataset: {name} (Source: {source_type}, ID: {identifier})")

    # --- Caching Logic ---
    # For yfinance, use time-based expiry.
    # For CSV, we might want to reload if the source CSV file has changed.
    # For simplicity now, we'll use CACHE_EXPIRY_DAYS_YFINANCE for yfinance
    # and a longer/different logic for CSV if desired, or just load CSV fresh each time if cache is disabled for it.
    
    cache_expiry = CACHE_EXPIRY_DAYS_YFINANCE if source_type == 'yfinance' else CACHE_EXPIRY_DAYS_CSV
    
    # Add more sophisticated cache check for CSV: e.g., compare source file mod time with cache mod time
    # For now, basic time expiry for all cached items.
    df = load_from_cache(name, expiry_days=cache_expiry)
    if df is not None:
        return df

    # --- Data Fetching/Loading ---
    if source_type == 'yfinance':
        start_date = dataset_config.get('start_date')
        end_date = dataset_config.get('end_date') # Typically not set, yfinance fetches up to latest
        data_fields = dataset_config.get('data_fields', ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
        df = fetch_from_yfinance(identifier, start_date, end_date, data_fields)
    
    elif source_type == 'csv':
        date_column = dataset_config.get('date_column')
        value_column = dataset_config.get('value_column') # For simple series like risk-free
        data_fields = dataset_config.get('data_fields') # For CSVs with multiple columns to select
        
        if not date_column:
            print(f"Error: 'date_column' not specified for CSV dataset {name}")
            return None
        # If data_fields is not specified, and value_column is, assume we want just that value_column as a series.
        # If data_fields is specified, it takes precedence.
        
        df = load_from_csv(identifier, date_column, value_column, data_fields)

    # elif source_type == 'fred': # Future extension
    #     series_id = dataset_config.get('identifier')
    #     # df = fetch_from_fred(series_id, ...)
    #     pass
        
    else:
        print(f"Warning: Unknown source_type '{source_type}' for dataset {name}")
        return None

    # --- Post-processing and Caching ---
    if df is not None and not df.empty:
        # Ensure DataFrame index is DatetimeIndex (should be handled by loaders)
        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"Warning: Index for {name} is not DatetimeIndex. Attempting conversion or check loader.")
            # Potentially convert or error, depending on strictness
        
        # Standardize column names if necessary (e.g. yfinance returns 'Adj Close', 'Volume', etc.)
        # df.columns = [col.title().replace(' ', '') for col in df.columns] # Example: Open, High, Low, Close, AdjClose, Volume

        save_to_cache(name, df)
        return df
    else:
        print(f"Failed to load data for {name}.")
        return None

def load_all_data(config_path=CONFIG_FILE) -> dict[str, pd.DataFrame]:
    """
    Loads all datasets defined in the configuration file.
    Returns a dictionary of DataFrames, keyed by dataset name.
    """
    dataset_configs = load_config(config_path)
    if not dataset_configs:
        print("Could not load dataset configurations. Exiting.")
        return {}

    all_data = {}
    for config in dataset_configs:
        dataset_name = config.get('name')
        if not dataset_name:
            print("Warning: Found a dataset entry without a 'name'. Skipping.")
            continue
        
        df = load_dataset(config)
        if df is not None:
            all_data[dataset_name] = df
            print(f"Successfully loaded and processed: {dataset_name}. Shape: {df.shape}")
        else:
            print(f"Failed to load or empty data for: {dataset_name}")
            
    print(f"\n--- Data loading summary ---")
    print(f"Successfully loaded {len(all_data)} out of {len(dataset_configs)} datasets.")
    for name, df in all_data.items():
        print(f"  - {name}: {df.shape[0]} rows, {df.shape[1]} columns. Date range: {df.index.min().date()} to {df.index.max().date()}")
    
    return all_data

# --- Main execution (for testing) ---
if __name__ == '__main__':
    print("--- Starting Data Loader Test ---")
    
    # 1. Ensure you have a 'datasets.yaml' file in the same directory.
    # 2. Ensure you have a 'data/risk_free.csv' (or as configured in datasets.yaml).
    # 3. This will create a 'cache/' directory.
    
    # Create dummy risk_free.csv if it doesn't exist for testing
    # (only if you don't have your actual risk_free.csv in data/ yet)
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    risk_free_path_in_yaml = None
    temp_configs = load_config() # to find the risk_free.csv path
    if temp_configs:
        for cfg in temp_configs:
            if cfg.get('name') == 'RiskFreeRate' and cfg.get('source_type') == 'csv':
                risk_free_path_in_yaml = Path(cfg.get('identifier'))
                break
    
    if risk_free_path_in_yaml and not risk_free_path_in_yaml.exists():
        print(f"Attempting to create a dummy '{risk_free_path_in_yaml.name}' for testing as it's missing...")
        dummy_csv_content = "DATE,DTB3\n2023-01-01,1.0\n2023-01-02,1.1\n2023-01-03,." # '.' for NaN example
        try:
            with open(risk_free_path_in_yaml, 'w') as f:
                f.write(dummy_csv_content)
            print(f"Dummy '{risk_free_path_in_yaml.name}' created in '{risk_free_path_in_yaml.parent}'.")
        except Exception as e:
            print(f"Could not create dummy csv: {e}")

    loaded_data = load_all_data()

    if loaded_data:
        print("\n--- Sample Data (first 3 rows of each loaded dataset) ---")
        for name, df in loaded_data.items():
            print(f"\nDataset: {name}")
            print(df.head(3))
            print("Info:")
            df.info()
    else:
        print("\nNo data was loaded.")
        
    print("\n--- Data Loader Test Complete ---")