import json
import datetime
import glob
import generate_stacks
import generate_grids
import generate_plots

# Settings from json config file
with open('config.json', 'r') as json_file:
    config = json.load(json_file)

if config["options"]["proc_step"] == 'stacking':
    generate_stacks.generate_stacks(config)

if config["options"]["proc_step"] == 'gridding':
    generate_grids.generate_grids(config)

if config["options"]["proc_step"] == 'plotting':
    generate_plots.generate_plots(config)
