import pandas as pd
import pycountry

def get_country_name(code):
    """Attempt to get the full country name from a 2-letter code."""
    try:
        # pycountry uses ISO 3166-1 alpha-2 (e.g., 'AR' -> 'Argentina')
        country = pycountry.countries.get(alpha_2=code.upper())
        if country:
            return country.name
            
        # If not found, check if it's a 3-letter code (alpha-3)
        country = pycountry.countries.get(alpha_3=code.upper())
        if country:
             return country.name
             
        # Specific overrides for non-standard codes EloRatings might use
        overrides = {
            'EN': 'England',
            'WA': 'Wales',
            'SQ': 'Scotland',
            'NI': 'Northern Ireland',
            'CW': 'Curaçao',
            'IR': 'Iran',
            'KR': 'South Korea',
        }
        return overrides.get(code.upper(), code) # Return original code if no match
        
    except Exception:
        return code

def download_and_format_elo_csv():
    url = "https://eloratings.net/2026_World_Cup.tsv"
    csv_filename = "2026_World_Cup_Elo_Ratings_Full_Names.csv"
    country_code_csv_filename = "2026_World_Cup_Elo_Ratings_Country_Codes.csv"
    
    print(f"Fetching raw data from {url}...")
    
    try:
        # Read the Tab-Separated web file directly into pandas
        df = pd.read_csv(url, sep='\t', header=None)
        
        # Add basic column names 
        column_count = len(df.columns)
        # We know Column 3 (index 2) is the abbreviation
        columns = [f"Column_{i+1}" for i in range(column_count)]
        columns[2] = "Country" # Rename it to be explicit
        df.columns = columns
        
        # Save to CSV
        df.to_csv(csv_filename, index=False)
        print(f"Success! Data saved to {csv_filename} with full country names.")
        
        # Save country codes to a separate CSV with names
        df_codes = df[["Country"]].copy()
        df_codes["Full_Name"] = df_codes["Country"].apply(get_country_name)
        df_codes.to_csv(country_code_csv_filename, index=False)
        print(f"Country codes and full names saved to {country_code_csv_filename}.")
        
    except Exception as e:
        print(f"Failed to fetch or save the data. Error: {e}")

if __name__ == "__main__":
    download_and_format_elo_csv()