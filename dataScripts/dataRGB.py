import ee
import geemap

# 1. Initialize (Authentication)
try:
    ee.Initialize(project='hale-life-482914-r0')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='hale-life-482914-r0')

print("✅ Authenticated to Google Earth Engine")

# --- REGION 1: CALIFORNIA (FIRE SCARS) ---
# Expanded coordinates to cover more of the Dixie & North Complex fires
# This is a large area, so it will generate MANY 256x256 chips (Great for training!)
roi_ca = ee.Geometry.Rectangle([-121.50, 39.50, -120.50, 40.50])

# Sentinel-2 for RGB (B4=Red, B3=Green, B2=Blue)
image_ca = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(roi_ca) \
    .filterDate('2023-06-01', '2023-09-01') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5)) \
    .median() \
    .select(['B4', 'B3', 'B2']) # <--- ONLY RGB

# Label for Fire: Hansen Global Forest Change
# 'lossyear' == 21 means loss occurred in 2021 (the big fire year there)
hansen = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')
loss_mask_ca = hansen.select('lossyear').eq(21).rename('label').toFloat()

# Stack and Export CA
final_stack_ca = image_ca.addBands(loss_mask_ca).toFloat()

task_ca = ee.batch.Export.image.toDrive(
    image=final_stack_ca,
    description='California_Fire_RGB_2023',
    folder='EcoRevive_RGB_Train/California', # It will create this subfolder
    scale=10, 
    region=roi_ca,
    fileFormat='GeoTIFF',
    maxPixels=1e10,
    fileDimensions=[256, 256], # Perfect for U-Net
    skipEmptyTiles=True
)
task_ca.start()
print("🚀 Started Task 1: California Fire (RGB)")


# --- REGION 2: LOUISIANA (WETLAND LOSS) ---
# Focusing on Terrebonne Bay area where water erosion is high
roi_la = ee.Geometry.Rectangle([-91.30, 29.20, -90.80, 29.60])

# Sentinel-2 for RGB
image_la = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(roi_la) \
    .filterDate('2023-01-01', '2023-12-31') \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5)) \
    .median() \
    .select(['B4', 'B3', 'B2'])

# Label for Wetlands: JRC Global Surface Water
# We look for pixels that transitioned from "Land" to "Water" (Loss)
water_jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
# 'transition' band: 3 = Land to Water (Permanent Loss)
loss_mask_la = water_jrc.select('transition').eq(3).rename('label').toFloat()

# Stack and Export LA
final_stack_la = image_la.addBands(loss_mask_la).toFloat()

task_la = ee.batch.Export.image.toDrive(
    image=final_stack_la,
    description='Louisiana_Wetland_RGB_2023',
    folder='EcoRevive_RGB_Train/Louisiana',
    scale=10, 
    region=roi_la,
    fileFormat='GeoTIFF',
    maxPixels=1e10,
    fileDimensions=[256, 256],
    skipEmptyTiles=True
)
task_la.start()
print("🚀 Started Task 2: Louisiana Wetlands (RGB)")

print("\n⚠️ Go to https://code.earthengine.google.com/tasks to monitor progress.")