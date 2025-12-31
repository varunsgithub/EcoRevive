import ee
import geemap

try:
    ee.Initialize(project='hale-life-482914-r0')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='hale-life-482914-r0')


## Plumas county -> California (Dixie fire)
roi_ca = ee.Geometry.Rectangle([-121.20, 39.80, -120.50, 40.50])

image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(roi_ca) \
    .filterDate('2024-06-01', '2024-09-01') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5)) \
    .median() \
    .select(['B4', 'B3', 'B2', 'B8']) 
    # Red, Green, Blue, NIR

# Get the Label (Hansen Global Forest Change)
# We want to find pixels that were lost specifically during the fire (2021)
hansen = ee.Image('UMD/hansen/global_forest_change_2024_v1_12')
# 'lossyear' is the year of loss (0-24). 21 = 2021.
# We create a binary mask: 1 = Burned in 2021, 0 = Other
loss_mask = hansen.select('lossyear').eq(21).rename('burned_label')

final_stack = image.addBands(loss_mask)

# Export to Google Drive
task = ee.batch.Export.image.toDrive(
    image=image.addBands(loss_mask),
    description='California_Dixie_Fire_2024',
    folder='Nature_Restoration_Project',
    scale=10, # 10 meter resolution
    region=roi_ca,
    fileFormat='GeoTIFF',
    maxPixels=1e10,
    fileDimensions=256,
    skipEmptyTiles=True
)

task.start()
print("Downloading.....")