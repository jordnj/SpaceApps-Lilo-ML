# Prerequisites:
# sudo apt install awscli
# aws configure 
# pip install boto3
from decimal import *
import boto3
from eolearn.core import EOTask, EOPatch, FeatureType,LoadTask, SaveTask, EOExecutor, ExtractBandsTask, MergeFeatureTask
import os
from statistics import mean
import numpy as np
from datetime import timezone, datetime
import glob
import copy

def create_location(lat, lon, time, note=""):
    return [{
        'lat': lat, # Should be a string
        'long': lon, # Should be a string
        'timestamp': time, # Should be an integer
        'description': note  # Should be a string
    }]
	
'''
# Example
date = '09-23-21' # Assign a string as date, should not contain / or \
location1 = create_location(38.12345, 66.8473, 1633278320)
location2 = create_location(76.123212, 76.343432, 1633278320)
debris = [location1, location2]

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('plastic-debris-table')

table.put_item(
    Item={
        'date': date,
        'plastic_cluster_data': debris
    }
)

# Use this code to retrieve data fro a specific date
response = table.get_item(
    Key={
        'date': date
    }
)

item = response['Item'] 
# Access the data by item["date"], item["plastic_cluster_data"], etc.
'''
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('plastic-debris-table')

outArr = [] #structs for eo patch
EOPATCH_FOLDER = os.path.join('.', 'eopatches')
directory_contents = os.listdir(EOPATCH_FOLDER)
for item in directory_contents:
	# load the EOPatch directory from local file
	patch = EOPatch.load(EOPATCH_FOLDER+"//"+item)
	
	# pull out debris, location, and timestamp data for the EOPatch
	classification = patch.data["CLASSIFICATION"]
	debris = classification[0] == 3
	debris = debris.astype(int)
	perc_debris = np.sum(debris.flatten())/len(debris)
	perc_debris = Decimal(perc_debris).quantize(Decimal('1.000')) # Round to the nearest 3 digits
	minlat = list(patch.bbox)[0]
	minlon = list(patch.bbox)[1]
	maxlat = list(patch.bbox)[2]
	maxlon = list(patch.bbox)[3]
	center_lat = Decimal(mean([minlat,maxlat])).quantize(Decimal('1.00000'))# Round to the nearest 5 digits
	center_lon = Decimal(mean([minlon,maxlon])).quantize(Decimal('1.00000'))# Round to the nearest 5 digits
	timestamp = int(patch.timestamp[0].replace(tzinfo=timezone.utc).timestamp())
	
	# prepare data for upload
	debris_data = create_location(str(center_lat), str(center_lon), timestamp, note="patch contains "+str(perc_debris)+" percent debris")
	outArr += copy.deepcopy(debris_data)

# upload data
date = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')

table.put_item(
Item={
    'date': date,
    'plastic_cluster_data': outArr
}
# delete the EOPatch from local filesystem
#shutil.rmtree(EOPATCH_FOLDER+"//"+item)
)
	
#clean up the scenes directory
#os.remove(glob.glob('.//scenes//*.json')[0])