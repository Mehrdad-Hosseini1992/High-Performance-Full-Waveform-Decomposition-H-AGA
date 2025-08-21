

import pandas as pd
from io import StringIO
import os
from typing import Union
from Utils.waveform import Waveform

class WaveformLoader:
    """
    Handles loading and parsing of waveform data from CSV files.
    """
    def load_csv(self, filepath: str) -> Union[Waveform, None]:
        """
        Loads CSV data with support for LeCroy oscilloscope format.
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Check for LeCroy format
            is_lecroy = any('LECROY' in line.upper() for line in lines[:5])
            
            if is_lecroy:
                print("  - Detected LeCroy format")
                
                # Find the CORRECT header line with Time and Ch1/Value
                header_idx = -1
                header_line = ""
                for i, line in enumerate(lines):
                    # Look specifically for the data header line
                    if 'Time' in line and ('Ch1' in line or 'Value' in line):
                        header_idx = i
                        header_line = line
                        print(f"  - Found data headers at line {i}: {line.strip()}")
                        break
                
                if header_idx != -1:
                    # Determine the correct delimiter
                    if '\t' in header_line:
                        delimiter = '\t'
                        print("  - Using tab delimiter")
                    elif ',' in header_line:
                        delimiter = ','
                        print("  - Using comma delimiter")
                    else:
                        delimiter = r'\s+'
                        print("  - Using whitespace delimiter")
                    
                    # Parse from the header line, skipping metadata lines
                    clean_lines = []
                    for i, line in enumerate(lines[header_idx:]):
                        # Skip empty lines and metadata
                        if line.strip() and not any(skip_word in line for skip_word in ['LECROY', 'Segment', 'Waveform', '#']):
                            clean_lines.append(line)
                    
                    data_string = "".join(clean_lines)
                    
                    # Parse with pandas
                    df = pd.read_csv(StringIO(data_string), sep=delimiter, engine='python', skipinitialspace=True)
                    
                    # Clean column names (remove empty columns)
                    df = df.dropna(axis=1, how='all')  # Drop columns that are all NaN
                    df.columns = [col.strip() for col in df.columns if col.strip()]
                    
                    print(f"  - Parsed {len(df)} data rows")
                    print(f"  - Detected columns: {df.columns.tolist()}")
                    
                else:
                    print("  - Could not find Time/Ch1 header, using fallback data detection...")
                    # Fallback: parse data lines directly
                    data_lines = []
                    for line in lines:
                        if ('E-' in line or 'e-' in line or '.' in line) and not any(word in line for word in ['LECROY', 'Segment', 'Waveform']):
                            # Try different delimiters
                            if '\t' in line:
                                parts = line.strip().split('\t')
                            elif ',' in line:
                                parts = line.strip().split(',')
                            else:
                                parts = line.strip().split()
                            
                            if len(parts) >= 2:
                                try:
                                    time_val = float(parts[0])
                                    ch1_val = float(parts[1])
                                    data_lines.append([time_val, ch1_val])
                                except ValueError:
                                    continue
                    
                    if data_lines:
                        df = pd.DataFrame(data_lines, columns=['Time', 'Ch1'])
                        print(f"  - Loaded {len(df)} data points using fallback method")
                    else:
                        print("  - No data found in file")
                        return None
                        
            else:
                # Standard CSV format (your original working code)
                header_idx = -1
                for i, line in enumerate(lines):
                    line_lower = line.lower()
                    if 'time' in line_lower and ('ch' in line_lower or 'average' in line_lower or 'ampl' in line_lower):
                        header_idx = i
                        break
                
                if header_idx == -1:
                    print("  - WARNING: Could not find a valid header. Attempting to read as a simple two-column CSV.")
                    df = pd.read_csv(filepath, header=None, usecols=[0, 1], on_bad_lines='skip', engine='python')
                    df.columns = ['Time', 'Ch1']
                else:
                    delimiter = ',' if ',' in lines[header_idx] else r'\s+'
                    data_string = "".join(lines[header_idx:])
                    df = pd.read_csv(StringIO(data_string), sep=delimiter, engine='python')

            # Clean up column names
            df.columns = [col.strip() for col in df.columns]
            
            # Handle 'Value' column (LeCroy sometimes uses this instead of Ch1)
            if 'Value' in df.columns:
                df.rename(columns={'Value': 'Ch1'}, inplace=True)
            
            # Remove unnamed/empty columns
            df = df.loc[:, df.columns.notna()]
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
            # Ensure we have the required columns
            if 'Time' not in df.columns:
                print(f"  - Warning: 'Time' column not found. Columns are: {df.columns.tolist()}")
                return None
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop NaN values in Time column
            df.dropna(subset=['Time'], inplace=True)
            
            if df.empty:
                print(f"  - No valid data points found in {os.path.basename(filepath)}")
                return None

            # Convert time to nanoseconds if needed
            if df['Time'].max() < 1e-6:
                print("  - Converting time from seconds to nanoseconds")
                df['Time'] = df['Time'] * 1e9

            print(f"  - Successfully loaded {len(df)} data points from {os.path.basename(filepath)}")
            print(f"  - Final columns: {df.columns.tolist()}")
            return Waveform(df)

        except Exception as e:
            print(f"  - Error loading {filepath}: {e}")
            import traceback
            traceback.print_exc()

            return None
