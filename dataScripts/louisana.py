import ee
import geemap

# Initialize
try:
    ee.Initialize(project='hale-life-482914-r0')
except:
    ee.Authenticate()
    ee.Initialize(project='hale-life-482914-r0')

# Region (Isle de Jean Charles / Terrebonne)
roi_la = ee.Geometry.Rectangle([-90.80, 29.10, -90.20, 29.60])

# Get the Input Image (Sentinel-2, Spring 2024)
# Spring is good for wetlands before summer algae blooms
image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(roi_la) \
    .filterDate('2024-03-01', '2024-05-01') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
    .median() \
    .select(['B4', 'B3', 'B2', 'B8'])

# Get the Label (JRC Global Surface Water)
# 'occurrence' is 0-100% water frequency over 38 years.
# High occurrence = Water (1), Low = Land (0).
water_hist = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
water_label = water_hist.select('occurrence').gt(80).rename('water_label')

final_stack = image.addBands(water_label)

# Export to Google Drive
task = ee.batch.Export.image.toDrive(
    image=final_stack,
    description='Louisiana_Wetland_2024',
    folder='Nature_Restoration_Project',
    scale=10,
    region=roi_la,
    fileFormat='GeoTIFF',
    maxPixels=1e10,
    fileDimensions=256,
    skipEmptyTiles=True
)

task.start()
print("Downloading....")