import os
import pandas as pd
import numpy as np
from IPython.display import display, HTML
import contextily as ctx
import matplotlib.pyplot as plt
import folium
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point
import math
import random
from rasterio.transform import from_origin
import rasterio
import sys
import json


# 0. set up directories
BASE_DIR = 'data/NEON'
SITE_CODE = 'HARV'
SITE_DIR = os.path.join(BASE_DIR, SITE_CODE)
SITE_IMAGE_DIR = os.path.join(SITE_DIR, "NEON_struct-plant")
PLOT_DIR = os.path.join(SITE_DIR, f"{SITE_CODE}_plots")
os.makedirs(PLOT_DIR, exist_ok=True)
PATCH_DIR = os.path.join(SITE_DIR, f"{SITE_CODE}_patch")
os.makedirs(PATCH_DIR, exist_ok=True)


class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # 保证及时写入
    def flush(self):
        for f in self.files:
            f.flush()

OUT_PATH = os.path.join(SITE_DIR, f"{SITE_CODE}_out.txt")
logfile = open(OUT_PATH, 'w')
sys.stdout = Tee(sys.__stdout__, logfile)
sys.stderr = Tee(sys.__stderr__, logfile)

def display_missing_values_info(df):
    print('======')
    print('shape:', df.shape)

    # Calculate the number of missing values and unique counts for each specified column
    missing_values = df.isnull().sum()
    unique_counts = df.nunique()

    # Display the missing values and unique counts
    print("Missing values and unique counts in the specified columns:")
    for column in df.columns:
        missing_count = missing_values[column]
        unique_count = unique_counts[column]
        # Align the output for better readability
        print(f"- {column:<16}: {missing_count:<10} missing values, {unique_count:<10} unique values")
    print('======')

    display(HTML(df.T.to_html(max_rows=60, max_cols=10)))

def process_vst_files(base_dir, file_identifier='vst_perplotperyear', columns=None):
    if columns is None:
        columns = [
            'eventID', 'uid', 'namedLocation', 'date', 'domainID', 
            'siteID', 'plotID', 'nlcdClass', 'decimalLatitude', 
            'decimalLongitude', 'easting', 'northing'
        ]

    sub_dirs = [entry.path for entry in os.scandir(base_dir) if entry.is_dir()]
    print("*** Start processing ***")
    print("Site code: ", SITE_CODE)
    print('total sub dirs:', len(sub_dirs))

    files_vst = []
    for sub_dir in sub_dirs:
        for root, _, files in os.walk(sub_dir):
            for file in files:
                if file_identifier in file:
                    file_path = os.path.join(root, file)
                    files_vst.append(file_path)

    all_dfs = [pd.read_csv(file_path) for file_path in files_vst]
    df_all = pd.concat(all_dfs, ignore_index=True)

    print("*** 'vst_perplotperyear' processing done ***")
    # display_missing_values_info(df_all)

    return df_all[columns]

def plot_vst_perplotperyear(df_vst_perplotperyear, SITE_CODE, PLOT_DIR):
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    plt.scatter(df_vst_perplotperyear['decimalLongitude'], df_vst_perplotperyear['decimalLatitude'], c='blue', s=10, alpha=0.7)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Plot Site {SITE_CODE} with OSM Background (WGS84)')
    plt.grid(True)
    
    # Add OSM basemap
    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik)
    save_path = os.path.join(PLOT_DIR, f"{SITE_CODE}_vst_perplotperyear.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"*** Plot saved to {save_path} ***")
    # plt.show()

def process_sub_dirs(base_dir):
    sub_dirs = [entry.path for entry in os.scandir(base_dir) if entry.is_dir()]
    df_mappingandtagging = []
    df_apparentindividual = []
    
    for sub_dir in sub_dirs:
        csv_files = [f for f in os.listdir(sub_dir) if f.endswith('.csv')]
        has_apparentindividual = False
        has_mappingandtagging = False
        
        for csv_file in csv_files:
            if 'apparentindividual' in csv_file:
                df = pd.read_csv(os.path.join(sub_dir, csv_file))
                df_apparentindividual.append(df)
                has_apparentindividual = True
            elif 'mappingandtagging' in csv_file:
                df = pd.read_csv(os.path.join(sub_dir, csv_file))
                df_mappingandtagging.append(df)
                has_mappingandtagging = True
        
        if not has_apparentindividual:
            print(f"No 'apparentindividual' file in {sub_dir}")
        if not has_mappingandtagging:
            print(f"No 'mappingandtagging' file in {sub_dir}")

    df_apparentindividual_combined = pd.concat(df_apparentindividual, ignore_index=True) if df_apparentindividual else None
    df_mappingandtagging_combined = pd.concat(df_mappingandtagging, ignore_index=True) if df_mappingandtagging else None
    print("*** 'mappingandtagging' processing done ***")
    # display_missing_values_info(df_mappingandtagging_combined)
    print("*** 'apparentindividual' processing done ***")
    # display_missing_values_info(df_apparentindividual_combined)

    return df_apparentindividual_combined, df_mappingandtagging_combined

def merge_ai_mt(df_apparentindividual_filtered, df_mappingandtagging_filtered):
    df_combined = pd.merge(
        df_apparentindividual_filtered,
        df_mappingandtagging_filtered,
        on='individualID',
        how='left',
        suffixes=('', '_drop')
    ).filter(regex='^(?!.*_drop)')
    df_combined = df_combined.dropna(subset=['stemDistance', 'stemAzimuth'])
    df_combined = df_combined.drop_duplicates(subset=['uid'])
    df_combined_filtered = df_combined[
        ['uid', 'namedLocation', 'date', 'eventID', 'domainID', 'siteID',
        'plotID', 'pointID', 'stemDistance', 'stemAzimuth', 
        'individualID', 'taxonID', 'scientificName', 'genus',
        'family', 'taxonRank', 'subplotID', 'tempStemID',
        'tagStatus', 'growthForm', 'plantStatus', 'stemDiameter',
        'height', 'maxCrownDiameter', 'ninetyCrownDiameter', 'canopyPosition']
    ]
    print("*** Merge height and genus processing done ***")
    # display_missing_values_info(df_combined_filtered)
    return df_combined_filtered

def merge_vst_perplotperyear(df_combined_filtered, df_vst_perplotperyear):
    df_all_unique = df_vst_perplotperyear.drop_duplicates(subset=['plotID'])
    df_combined_georeferenced = pd.merge(
        df_combined_filtered,
        df_all_unique[['plotID', 'decimalLatitude', 'decimalLongitude', 'easting', 'northing']],
        on='plotID',
        how='left'
    )
    print("*** Merge vst_perplotperyear processing done ***")
    return df_combined_georeferenced

def pol2latlon(lat, lon, distance, azimuth):
    R = 6378137
    # lat, lon: degrees
    # distance: meters
    # azimuth: degrees, 0 is north, clockwise
    lat1 = np.deg2rad(lat)
    lon1 = np.deg2rad(lon)
    az = np.deg2rad(azimuth)
    d = distance

    lat2 = np.arcsin(np.sin(lat1) * np.cos(d / R) + np.cos(lat1) * np.sin(d / R) * np.cos(az))
    lon2 = lon1 + np.arctan2(np.sin(az) * np.sin(d / R) * np.cos(lat1), np.cos(d / R) - np.sin(lat1) * np.sin(lat2))
    return np.rad2deg(lat2), np.rad2deg(lon2)

def add_tree_lat_lon(df):
    # Calculate latitude and longitude for each tree
    lat, lon = pol2latlon(
        df['decimalLatitude'].astype(float).values,
        df['decimalLongitude'].astype(float).values,
        df['stemDistance'].astype(float).values,
        df['stemAzimuth'].astype(float).values
    )
    df['tree_lat'] = lat
    df['tree_lon'] = lon
    
    # Calculate easting and northing for each tree
    easting, northing = df['easting'].values, df['northing'].values
    tree_easting = easting + df['stemDistance'].astype(float).values * np.sin(np.deg2rad(df['stemAzimuth'].astype(float).values))
    tree_northing = northing + df['stemDistance'].astype(float).values * np.cos(np.deg2rad(df['stemAzimuth'].astype(float).values))
    
    df['tree_easting'] = tree_easting
    df['tree_northing'] = tree_northing
    print("*** Add tree lat and lon processing done ***")
    return df

def plot_and_save_tree_distribution(df, SITE_CODE, PLOT_DIR):
    # Plot all points with x and y axis labeled with coordinates
    plt.figure(figsize=(8, 8))
    plt.scatter(df['tree_easting'], df['tree_northing'], s=0.7, alpha=1, color='blue')
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    plt.title(f'{SITE_CODE} Processed Tree Points Distribution ({len(df)} points)')

    # Calculate and print the coordinate distribution range and dimensions
    easting_min, easting_max = df['tree_easting'].min(), df['tree_easting'].max()
    northing_min, northing_max = df['tree_northing'].min(), df['tree_northing'].max()
    print(f"Easting range: {easting_min:.2f} m ~ {easting_max:.2f} m")
    print(f"Northing range: {northing_min:.2f} m ~ {northing_max:.2f} m")

    width_m = easting_max - easting_min
    height_m = northing_max - northing_min
    print(f"Area dimensions: {width_m:.2f} m × {height_m:.2f} m")

    plt.xlim(easting_min, easting_max)
    plt.ylim(northing_min, northing_max)
    plt.grid(True, linestyle='--', alpha=0.5)
    filename = f'{SITE_CODE}_processed_tree_distribution.png'
    plt.savefig(os.path.join(PLOT_DIR, filename))
    # plt.show()

def check_res(df, SITE_CODE, PLOT_DIR):
    # Calculate center point
    center_lat = df['tree_lat'].mean()
    center_lon = df['tree_lon'].mean()

    # Convert tree lat/lon to GeoDataFrame
    geometry = [Point(xy) for xy in zip(df['tree_lon'], df['tree_lat'])]
    gdf_trees = gpd.GeoDataFrame(df, geometry=geometry)

    # Calculate bounding box for the entire area
    min_lat, max_lat = df['tree_lat'].min(), df['tree_lat'].max()
    min_lon, max_lon = df['tree_lon'].min(), df['tree_lon'].max()

    # Create folium map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles='OpenStreetMap', max_zoom=22)

    # Add bounding box to map
    folium.Rectangle(
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        color='red',
        fill=True,
        fill_opacity=0
    ).add_to(m)

    # Calculate 10m grid
    lat_grid_offset = 60 / 111000
    lon_grid_offset = 60 / (111000 * abs(math.cos(math.radians(center_lat))))

    # Draw 10m grid
    lat_lines = int((max_lat - min_lat) / lat_grid_offset)
    lon_lines = int((max_lon - min_lon) / lon_grid_offset)

    for i in range(lat_lines + 1):
        lat = min_lat + i * lat_grid_offset
        folium.PolyLine(
            locations=[(lat, min_lon), (lat, max_lon)],
            color='green',
            weight=0.1,
            opacity=0.5
        ).add_to(m)

    for j in range(lon_lines + 1):
        lon = min_lon + j * lon_grid_offset
        folium.PolyLine(
            locations=[(min_lat, lon), (max_lat, lon)],
            color='green',
            weight=0.1,
            opacity=0.5
        ).add_to(m)

    # Filter trees within the entire area
    gdf_trees_in_area = gdf_trees.cx[min_lon:max_lon, min_lat:max_lat]

    # Randomly select 500 tree points
    sampled_geometries = random.sample(list(gdf_trees_in_area.geometry), min(500, len(gdf_trees_in_area)))

    # Integrate tqdm into loop
    for geom, row in tqdm(zip(sampled_geometries, gdf_trees_in_area.itertuples()), total=len(sampled_geometries), desc='Adding tree points'):
        folium.CircleMarker(
            location=[row.tree_lat, row.tree_lon],
            radius=5,  # Small radius to represent a point
            color='yellow',  # Color of the point
            fill=True,
            fill_opacity=1,
            popup=f"Lat: {row.tree_lat}, Lon: {row.tree_lon}, Genus: {row.genus}"  # Popup with tree_lat, tree_lon, genus
        ).add_to(m)
    m.save(os.path.join(PLOT_DIR, f'{SITE_CODE}_tree_distribution.html'))
    # Display map
    print(f'*** {SITE_CODE} tree distribution map saved to {PLOT_DIR} ***')
    return m

def generate_patch_tiles(df, genus_encoding, patch_dir, site_code,
                         patch_size=100, stride=100, resolution=1, crs='EPSG:32618'):

    # Filter valid points
    df_valid = df.dropna(subset=['tree_easting', 'tree_northing', 'height', 'genus'])
    # Compute global extents
    min_e, max_e = df_valid['tree_easting'].min(), df_valid['tree_easting'].max()
    min_n, max_n = df_valid['tree_northing'].min(), df_valid['tree_northing'].max()

    patch_id = 0

    for x0 in tqdm(np.arange(min_e, max_e, stride), desc="Processing Easting"):
        for y0 in tqdm(np.arange(min_n, max_n, stride), desc="Processing Northing", leave=False):
            # Select points in current window
            mask = (
                (df_valid['tree_easting'] >= x0) & (df_valid['tree_easting'] < x0 + patch_size) &
                (df_valid['tree_northing'] >= y0) & (df_valid['tree_northing'] < y0 + patch_size)
            )
            pts = df_valid[mask]
            if pts.empty:
                continue

            # Initialize rasters
            height_arr = np.zeros((patch_size, patch_size), dtype=np.float32)
            genus_arr  = np.zeros((patch_size, patch_size), dtype=np.float32)
            # Rasterize points
            for _, row in pts.iterrows():
                ix = int(row['tree_easting'] - x0)
                iy = int(row['tree_northing'] - y0)
                iy = patch_size - 1 - iy  # Flip y for raster row index
                if 0 <= ix < patch_size and 0 <= iy < patch_size:
                    height_arr[iy, ix] = row['height']
                    genus_arr[iy, ix]  = genus_encoding[row['genus']]

            # Mask invalid values
            height_arr = np.ma.masked_equal(height_arr, 0)
            genus_arr = np.ma.masked_equal(genus_arr, 0)

            # Write GeoTIFF
            transform = from_origin(x0, y0 + patch_size, resolution, resolution)
            out_fp = os.path.join(patch_dir, f"patch_{patch_id:05d}.tif")
            with rasterio.open(
                out_fp, 'w',
                driver='GTiff',
                height=patch_size,
                width=patch_size,
                count=2,
                dtype=height_arr.dtype,
                crs=crs,
                transform=transform,
                nodata=0
            ) as dst:
                dst.write(height_arr.filled(0), 1)  # Use filled to write masked arrays
                dst.write(genus_arr.filled(0), 2)

            patch_id += 1
    print(f"*** {patch_id} patches saved to {patch_dir} ***")

    return patch_id

def plot_patch_tiles(patch_dir, SITE_CODE, PLOT_DIR, start_patch_id=55, end_patch_id=64, figsize=(10, 10)):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=figsize)
    axes = axes.flatten()

    for i, patch_id in enumerate(range(start_patch_id, end_patch_id)):
        with rasterio.open(f'{patch_dir}/patch_{patch_id:05d}.tif') as src:
            band = src.read(1)
            # Mask out the zero values
            band = np.ma.masked_equal(band, 0)
            # Get the transform to calculate coordinates
            transform = src.transform
        ax = axes[i]
        cax = ax.imshow(band, cmap='viridis')
        ax.set_title(f'{SITE_CODE} Patch {patch_id:05d}')
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        # Calculate and display coordinates
        x_min, y_max = transform * (0, 0)
        x_max, y_min = transform * (band.shape[1], band.shape[0])
        ax.set_xlabel(f'X: {x_min:.1f} to {x_max:.1f}')
        ax.set_ylabel(f'Y: {y_min:.1f} to {y_max:.1f}')
        
        # Add 10m grid
        ax.set_xticks(np.arange(0, band.shape[1], 10))
        ax.set_yticks(np.arange(0, band.shape[0], 10))
        ax.grid(which='both', color='gray', linestyle='--', linewidth=0.5)

    # Create a new axis for the colorbar outside the main plot
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(cax, cax=cbar_ax, orientation='vertical', label='Height')
    fig.savefig(os.path.join(PLOT_DIR, f'{SITE_CODE}_patch_tiles.png'))
    print(f'*** {SITE_CODE} patch tiles saved to {PLOT_DIR} ***')
    # plt.show()

def plot_patch_height_genus(
    SITE_CODE,
    PLOT_DIR,
    patch_dir,
    genus_encoding,
    figsize=(16, 6),
    grid_interval=10
):
    file_path = os.path.join(patch_dir, "patch_00055.tif")
    with rasterio.open(file_path) as src:
        height = src.read(1)
        genus = src.read(2)
        
        # Mask out the zero values for better visualization
        height = np.ma.masked_equal(height, 0)
        genus = np.ma.masked_equal(genus, 0)
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        
        # Plot height
        ax1 = axes[0]
        cax1 = ax1.imshow(height, cmap='viridis')
        ax1.set_title(f'Height {len(height)} points')
        fig.colorbar(cax1, ax=ax1, orientation='vertical', label='Height')
        ax1.set_xticks(np.arange(0, height.shape[1], grid_interval))
        ax1.set_yticks(np.arange(0, height.shape[0], grid_interval))
        ax1.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
        
        # Plot genus
        ax2 = axes[1]
        cax2 = ax2.imshow(genus, cmap='plasma')
        ax2.set_title(f'Genus {len(genus)} points')
        ax2.set_xticks(np.arange(0, genus.shape[1], grid_interval))
        ax2.set_yticks(np.arange(0, genus.shape[0], grid_interval))
        ax2.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
        
        # Create a legend for genus and place it beside the plot
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', label=genus_name,
                       markerfacecolor=plt.cm.plasma(i/len(genus_encoding)), markersize=10)
            for i, genus_name in enumerate(genus_encoding.keys())
        ]
        ax2.legend(handles=handles, title='Genus', loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(os.path.join(PLOT_DIR, f'{SITE_CODE}_patch55_height_genus.png'))
        print(f'*** {SITE_CODE} patch55 height and genus saved to {PLOT_DIR} ***')
        plt.tight_layout()
        # plt.show()

def __main__():
    # 1. process vst files
    df_vst_perplotperyear = process_vst_files(SITE_IMAGE_DIR)

    # 2. process apparentindividual files
    plot_vst_perplotperyear(df_vst_perplotperyear, SITE_CODE, PLOT_DIR)

    # 3. process sub dirs
    df_apparentindividual_combined, df_mappingandtagging_combined = process_sub_dirs(SITE_IMAGE_DIR)

    # 4. filter data
    df_mappingandtagging_filtered = df_mappingandtagging_combined.dropna(subset=['stemDistance', 'stemAzimuth', 'genus'])
    df_apparentindividual_filtered = df_apparentindividual_combined
    df_combined_filtered = merge_ai_mt(df_apparentindividual_filtered, df_mappingandtagging_filtered)

    # 5. merge vst_perplotperyear
    df_combined_georeferenced = merge_vst_perplotperyear(df_combined_filtered, df_vst_perplotperyear)

    # 6. add tree lat and lon
    df_combined_georeferenced = add_tree_lat_lon(df_combined_georeferenced)
    df_combined_georeferenced_sorted = df_combined_georeferenced.sort_values(by='date', ascending=False)
    df_combined_georeferenced_unique = df_combined_georeferenced_sorted.drop_duplicates(subset=['tree_lat', 'tree_lon'])
    df_combined_georeferenced = df_combined_georeferenced_unique
    print("*** Drop duplicate tree lat and lon processing done ***")
    print("*** Useful points: valid genus but some nan in height ***")
    # display_missing_values_info(df_combined_georeferenced)

    # 7. plot and save tree distribution
    plot_and_save_tree_distribution(df_combined_georeferenced, SITE_CODE, PLOT_DIR)
    m = check_res(df_combined_georeferenced, SITE_CODE, PLOT_DIR)
   
    # 8. generate patch tiles
    genus_unique_sorted = sorted(df_combined_georeferenced['genus'].unique())
    genus_encoding = {genus: idx + 1 for idx, genus in enumerate(genus_unique_sorted)}
    print("======")
    print(f"*** Start generating patch tiles, total {len(genus_encoding)} genus ***")

    patch_id = generate_patch_tiles(df_combined_georeferenced, genus_encoding, PATCH_DIR, SITE_CODE)

    # 9. plot patch tiles
    plot_patch_tiles(PATCH_DIR, SITE_CODE, PLOT_DIR)
    plot_patch_height_genus(SITE_CODE, PLOT_DIR, PATCH_DIR, genus_encoding)

if __name__ == "__main__":
    __main__()