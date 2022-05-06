

import subprocess
import glob

script = '/home/bozcomlekci/anaconda3/envs/op/bin/gdal_polygonize.py'
for in_file in glob.glob("lulc/bayindir_20_21.tif"):
    out_file = in_file[:-4] + ".json"
    subprocess.call(["python",script,in_file,'-f','GeoJSON',out_file])