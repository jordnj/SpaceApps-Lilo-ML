# Built-in modules
import pickle
import sys
import os
import datetime
import itertools
from aenum import MultiValueEnum

# Basics of Python data handling and visualization
import numpy as np
np.random.seed(42)
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon
from tqdm.auto import tqdm
import glob
import json

# Imports from eo-learn and sentinelhub-py
from eolearn.core import EOTask, EOPatch, LinearWorkflow, FeatureType, OverwritePermission, \
    LoadTask, SaveTask, EOExecutor, ExtractBandsTask, MergeFeatureTask
from eolearn.io import SentinelHubInputTask, VectorImportTask, ExportToTiffTask
from eolearn.mask import AddValidDataMaskTask
from eolearn.geometry import VectorToRasterTask, PointSamplingTask, ErosionTask
from eolearn.features import LinearInterpolationTask, SimpleFilterTask, NormalizedDifferenceIndexTask
from sentinelhub import UtmZoneSplitter, BBox, CRS, DataCollection

import geopandas as gp
from shapely.geometry import box
from sentinelhub import UtmZoneSplitter, BBoxSplitter

DEFAULT_BAND_NAMES= ['B01', 
                 'B02', 
                 'B03', 
                 'B04', 
                 'B05', 
                 'B06', 
                 'B07', 
                 'B08', 
                 'B08A', 
                 'B09', 
                 'B10', 
                 'B11', 
                 'B12']

class CalcFDI(EOTask):
    ''' EOTask that calculates the floating debris index see https://www.nature.com/articles/s41598-020-62298-z

        Expectes the EOPatch to have either Sentinel L1C or 
        L2A bands.

        Will append the data layer "FDI" to the EOPatch

        Run time parameters:
            - band_layer(str): the name of the data layer to use for raw Sentinel bands
            - band_names(str): the names of each band B01, B02 etc
    '''
    
    def execute(self,
                eopatch,
                band_layer='BANDS-S2-L1C', 
                band_names=DEFAULT_BAND_NAMES
                ):
        bands  = eopatch[FeatureType.DATA][band_layer]
        NIR = bands[:,:,:,band_names.index('B08')]
        RE  = bands[:,:,:,band_names.index('B05')]
        SWIR = bands[:,:,:,band_names.index('B11')]
        factor = 1.636
        debris_index = NIR - ( RE + np.multiply(SWIR - RE, factor) )
        FDI_calculation = debris_index.reshape([bands.shape[0], bands.shape[1], bands.shape[2], 1])
        
        eopatch[(FeatureType.DATA,'FDI')] = FDI_calculation
        return eopatch
class WaterDetector(EOTask):
    """
        Very simple water detector based on NDWI threshold.

        Adds the mask layer "WATER_MASK" to the EOPatch.

        Expects the EOPatch to have an "NDWI" layer.

        Run time arguments:
            - threshold(float): The cutoff threshold for water.
        
    """
    
    @staticmethod
    def detect_water(ndwi,threshold):  
        return ndwi > threshold

    def execute(self, eopatch, threshold=0.5):
        band_layer='BANDS-S2-L1C'
        water_masks = np.asarray([self.detect_water(ndwi[...,0], threshold) for ndwi in eopatch[FeatureType.DATA][band_layer]])
        eopatch[(FeatureType.MASK,'WATER_MASK')] = water_masks.reshape([water_masks.shape[0], water_masks.shape[1], water_masks.shape[2], 1])
        return eopatch
class CombineMask(EOTask):
    ''' Simple task to combine the various masks in to one

        Run time parameters passed on workflow execution: 
        use_water(bool): Include the water mask as part of the full mask. Default is false
    '''
    
    def execute(self,eopatch, use_water=True):
        if(use_water):
            combined = np.logical_or( np.invert(eopatch[FeatureType.MASK]['WATER_MASK']).astype(bool),eopatch[FeatureType.MASK]['CLM'].astype(bool) )
        else:
            combined = eopatch[FeatureType.MASK]['CLM'].astype(bool)
        eopatch[(FeatureType.MASK,'FULL_MASK')] = combined #np.invert(eopatch.mask['WATER_MASK']) & eopatch.mask['CLM'] & np.invert(eopatch.mask['IS_DATA']))
        return eopatch

# Locations for collected data and intermediate results
EOPATCH_FOLDER = os.path.join('.', 'eopatches')
RESULTS_FOLDER = os.path.join('.', 'results')
os.makedirs(EOPATCH_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

#Define Workflow Tasks
band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
add_data = SentinelHubInputTask(
    bands_feature=(FeatureType.DATA, 'BANDS-S2-L1C'),
    bands = band_names,
    resolution=10,
    maxcc=0.8,
    time_difference=datetime.timedelta(minutes=120),
    data_collection=DataCollection.SENTINEL2_L1C,
    additional_data=[(FeatureType.MASK, 'dataMask', 'IS_DATA'),
                     (FeatureType.MASK, 'CLM')]
)
add_l2a = SentinelHubInputTask(
    data_collection=DataCollection.SENTINEL2_L2A,
    bands=['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12'],
    bands_feature=(FeatureType.DATA, 'BANDS-S2-L2A'),
    resolution=10,
    maxcc=0.8,
    time_difference=datetime.timedelta(minutes=120)
)
# CALCULATING NEW FEATURES
# NDVI: (B08 - B04)/(B08 + B04)
# NDWI: (B03 - B08)/(B03 + B08)
# NDBI: (B11 - B08)/(B11 + B08)
ndvi = NormalizedDifferenceIndexTask((FeatureType.DATA, 'BANDS-S2-L1C'), (FeatureType.DATA, 'NDVI'),
                                     [band_names.index('B08'), band_names.index('B04')])
ndwi = NormalizedDifferenceIndexTask((FeatureType.DATA, 'BANDS-S2-L1C'), (FeatureType.DATA, 'NDWI'),
                                     [band_names.index('B03'), band_names.index('B08')])
ndbi = NormalizedDifferenceIndexTask((FeatureType.DATA, 'BANDS-S2-L1C'), (FeatureType.DATA, 'NDBI'),
                                     [band_names.index('B11'), band_names.index('B08')])
add_fdi = CalcFDI()
water_mask = WaterDetector()
combined_masks = CombineMask()
# SAVE TO OUTPUT (if needed)
save = SaveTask(EOPATCH_FOLDER, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

EOPATCH_FOLDER = os.path.join('.', 'eopatches')

# Define the workflow
workflow = LinearWorkflow(
    add_data,
	add_l2a,
    ndvi,
    ndwi,
	add_fdi,
	water_mask,
	combined_masks,
    save
)

json_file = glob.glob('scenes//*.json')[0]
f = open(json_file)
scene = json.load(f)
time_interval = scene['timeRange']
minLon = scene['minLon']
maxLon = scene['maxLon']
minLat = scene['minLat']
maxLat = scene['maxLat']

# Time interval for the SH request
#time_interval = ['2019-04-28', '2019-04-29']#["2018-10-30","2018-11-01"]#

#Durban Harbor
#minLon = -29.908949
#maxLon = -29.850490
#minLat = 30.991053
#maxLat = 31.079215
region = box(minLat,minLon,maxLat,maxLon)
bbox_splitter = BBoxSplitter([region],CRS.WGS84, (10, 10) )

bbox_list = np.array(bbox_splitter.get_bbox_list())
info_list = np.array(bbox_splitter.get_info_list())

# Prepare info of selected EOPatches
geometry = [Polygon(bbox.get_polygon()) for bbox in bbox_list]
idxs_x = [info['index_x'] for info in info_list]
idxs_y = [info['index_y'] for info in info_list]
ids = range(len(info_list))

gdf = gp.GeoDataFrame({'index': ids,'index_x': idxs_x, 'index_y': idxs_y},
                crs='EPSG:4326',
                geometry=geometry)

for idx,row in gdf.to_crs("EPSG:3857").iterrows():
        geo = row.geometry
        xindex = row['index_x']
        yindex = row['index_y']
        index = row['index']

# Define additional parameters of the workflow
execution_args = []
for idx, bbox in enumerate(bbox_list):
    execution_args.append({
        add_data: {'bbox': bbox, 'time_interval': time_interval},
        save: {'eopatch_folder': f'eopatch_{idx}'}
    })

# Execute the workflow
executor = EOExecutor(workflow, execution_args, save_logs=True)
executor.run(workers=5, multiprocess=True)

failed_ids = executor.get_failed_executions()
if failed_ids:
    raise RuntimeError(f'Execution failed EOPatches with IDs:\n{failed_ids}\n'
                       f'For more info check report at {executor.get_report_filename()}')