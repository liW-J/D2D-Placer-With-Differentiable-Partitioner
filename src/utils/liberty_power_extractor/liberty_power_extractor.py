'''
Author: JeanneWillis hi@jeannewillis.cn
Date: 2025-03-17 16:03:18
LastEditors: JeanneWillis hi@jeannewillis.cn
LastEditTime: 2025-03-17 17:28:47
FilePath: /D2D-placer/src/utils/liberty_power_extractor.py
Description: extractor area-power map
'''

def extract_area_power_relation(lib_file_path):
    area_power_dict = {}
    current_cell = None
    
    with open(lib_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # find Modlule
            if line.startswith('cell ('):
                current_cell = line.split('(')[1].split(')')[0].strip()
                area_power_dict[current_cell] = {'area': None, 'leakage_power': None}
            
            if current_cell:
                # find area
                if 'area' in line and ':' in line:
                    area = float(line.split(':')[1].strip().rstrip(';'))
                    area_power_dict[current_cell]['area'] = area
                
                # find cell_leakage_power
                if 'cell_leakage_power' in line and ':' in line:
                    leakage = float(line.split(':')[1].strip().rstrip(';'))
                    area_power_dict[current_cell]['leakage_power'] = leakage

    return area_power_dict

def print_results(area_power_dict):

    print("\nArea and Power Relationship:")
    print("=" * 60)
    print(f"{'Cell Name':<20} {'Area':<15} {'Leakage Power':<15}")
    print("-" * 60)
    
    for cell, data in area_power_dict.items():
        area = data['area']
        power = data['leakage_power']
        if area is not None and power is not None:
            print(f"{cell:<20} {area:<15.6f} {power:<15.6f}")
    print("=" * 60)

def main():
  
    lib_file = "nangate45/1NangateOpenCellLibrary_typical.lib"
    
    results = extract_area_power_relation(lib_file)
    
    # print_results(results)
    
    # save to csv
    import pandas as pd
    df = pd.DataFrame.from_dict(results, orient='index')
    df = df.sort_values('area')
    df.to_csv('area_power_relation.csv')

if __name__ == "__main__":
    main()