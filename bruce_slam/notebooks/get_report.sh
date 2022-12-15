sim_scene="rfal_land"

python batch_report.py "$sim_scene" 1 1
python heat_map.py "$sim_scene"
python box_plot.py "$sim_scene"

