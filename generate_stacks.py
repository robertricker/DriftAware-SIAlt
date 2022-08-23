import datetime
import stacking_lib
import gridding_lib
import glob


def generate_stacks(config):
    if config["options"]["sensor"] == 'is2':
        is2_hem = "01" if config["options"]["hem"] == "nh" else "02"
        source_list = sorted(glob.glob(config["dir"]["is2_source"] + "*/*" + is2_hem + '*.h5'))
        out_dir = config["dir"]["is2_geojson"]
    else:
        source_list = sorted(glob.glob(config["dir"]["cs2_source"] + '*/*/*/*.nc'))
        out_dir = config["dir"]["cs2_geojson"]

    osi405_list = sorted(glob.glob(config["dir"]["osi405"] + "*.nc"))
    osi430b_list = sorted(glob.glob(config["dir"]["osi430b"] + "*.nc"))

    grid, cell_width = gridding_lib.define_grid(
        config["stack_grid"]["bounds"][0],
        config["stack_grid"]["bounds"][1],
        config["stack_grid"]["bounds"][2],
        config["stack_grid"]["bounds"][3],
        config["stack_grid"]["dim"],
        config["stack_grid"]["epsg"])

    stacking_lib.stack_proc(
        datetime.datetime.strptime(config["options"]["t_start"], '%Y-%m-%d %H:%M:%S'),
        config["options"]["t_window_length"],
        config["options"]["t_series_length"],
        config["options"]["hem"],
        config["options"]["sensor"],
        source_list,
        osi405_list,
        osi430b_list,
        1,
        grid,
        out_dir)

    stacking_lib.stack_proc(
        datetime.datetime.strptime(config["options"]["t_start"], '%Y-%m-%d %H:%M:%S'),
        config["options"]["t_window_length"],
        config["options"]["t_series_length"],
        config["options"]["hem"],
        config["options"]["sensor"],
        source_list,
        osi405_list,
        osi430b_list,
        -1,
        grid,
        out_dir)

    list_f = sorted(glob.glob(out_dir + '*-f-*.geojson'))
    list_r = sorted(glob.glob(out_dir + '*-r-*.geojson'))

    stacking_lib.merge_forward_reverse_stacks(
        datetime.datetime.strptime(config["options"]["t_start"], '%Y-%m-%d %H:%M:%S'),
        config["options"]["t_series_length"],
        list_f,
        list_r,
        out_dir)

