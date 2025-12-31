import ee
import geemap

try:
    ee.Initialize(project='hale-life-482914-r0')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='hale-life-482914-r0')

# Plumas County / Dixie Fire Region
roi_ca = ee.Geometry.Rectangle([-121.20, 39.80, -120.50, 40.50])

# Input Image (Now with 10 bands - excellent choice)
image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(roi_ca) \
    .filterDate('2024-06-01', '2024-09-01') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5)) \
    .median() \
    .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']) 

# Label (Forest Loss)
hansen = ee.Image('UMD/hansen/global_forest_change_2024_v1_12')

loss_mask = hansen.select('lossyear').eq(21).rename('burned_label').toFloat()

# Create Stack
final_stack = image.addBands(loss_mask).toFloat()

# Export
task = ee.batch.Export.image.toDrive(
    image=final_stack,  # Use the stack variable here
    description='California_Dixie_Fire_2024_Full',
    folder='Nature_Restoration_California',
    scale=10, 
    region=roi_ca,
    fileFormat='GeoTIFF',
    maxPixels=1e10,
    fileDimensions=[256, 256],
    skipEmptyTiles=True
)

task.start()
print("Downloading... Check https://code.earthengine.google.com/tasks")