"""
Script to create MOGREPS-UK cross-section plots.
"""
import numpy as np
import os
import sys
import glob
import subprocess
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
from datetime import datetime, timedelta
from dateutil.rrule import rrule, MINUTELY, HOURLY
import iris
import iris.plot as iplt
from iris.analysis.cartography import rotate_pole
from multiprocessing import Process, Queue
# Local import
import useful_functions as uf

# Import environment constants
try:
    USER = os.environ['USER']
    HPC_DIR = os.environ['MOG_UK_DIR']
    SCRATCH_DIR = os.environ['SCRATCH_DIR']
    HTML_DIR = os.environ['HTML_DIR']
    URL_START = os.environ['URL_START']
    SIDEBAR = os.environ['SIDEBAR']
except KeyError as err:
    raise IOError(f'Environment variable {str(err)} not set.')

# Iris constraints
U_CON = iris.AttributeConstraint(STASH='m01s00i002')
V_CON = iris.AttributeConstraint(STASH='m01s00i003')
OROG_CON = iris.AttributeConstraint(STASH='m01s00i033')
TEMP_1P5_CON = iris.AttributeConstraint(STASH='m01s03i236')
TEMP_CON = iris.AttributeConstraint(STASH='m01s16i004')
SPEC_HUM_CON = iris.AttributeConstraint(STASH='m01s00i010')
PRES_CON = iris.AttributeConstraint(STASH='m01s00i408')
REL_HUM_CON = iris.AttributeConstraint(STASH='m01s03i245')
RAIN_CON = iris.AttributeConstraint(STASH='m01s04i203')
VIS_CON = iris.AttributeConstraint(STASH='m01s03i281')

# ==============================================================================
# Change these bits for new trial site/date
# Dates of start and end of trial
FIRST_DT = datetime(2022, 5, 2, 0)  # Year, month, day, hour
LAST_DT = datetime(2022, 5, 3, 1)  # Year, month, day, hour
# Location/height/name of site
LAT = 53.225556
LON = -0.881389
SITE_HEIGHT = 43
SITE_NAME = 'National Rail'
# ==============================================================================

# Image directory
IMG_DIR = f'{HTML_DIR}/images/{SITE_NAME.replace(" ", "_")}'
# Lead time numbers used in filenames
FNAME_NUMS = [str(num).zfill(3) for num in range(0, 126, 3)]
# For converting mph to knots
MPH_TO_KTS = 0.86897423357831
# Threshold lists (wind thresholds need to be in knots as well as mph)
WIND_THRESHS = [12, 15, 20, 25]
TEMP_THRESHS = [0, 20, 25, 30]
REL_HUM_THRESHS = [40, 95]
RAIN_THRESHS = [0.2, 1., 4.]
VIS_THRESHS = [10000, 5000, 1000, 500, 200]
# Ratio of molecular weights of water and air
REPSILON = 0.62198


def main(new_data):
    """
    Copies files from HPC, extracts data and creates wind, temperature and
    relative humidity cubes. Probabilities are then calculated based on a few
    thresholds and cross-section plots are created and saved as png files.
    HTML page displaying plots is also updated.
    """
    # Make image directory if needed
    if not os.path.exists(IMG_DIR):
        os.system(f'mkdir {IMG_DIR}')

    # Get some required variables based on forecast period
    rec_m_date, start_vdt, end_vdt = get_vrbs()

    # Remove old MOGREPS-UK files
    housekeep(rec_m_date)

    # If new data required, extract cubes from MOGREPS-UK files on HPC
    if new_data == 'yes':
        (wind_spd_cube_list, temp_cube_list, rel_hum_cube_list,
         rain_cube_list, vis_cube_list) = get_cubes(rec_m_date, start_vdt,
                                                    end_vdt)

    # Otherwise (for testing), latest pickled data can be used
    else:
        (wind_spd_cube_list, temp_cube_list, rel_hum_cube_list,
         rain_cube_list,
         vis_cube_list) = uf.unpickle_data(f'{SCRATCH_DIR}/pickle')

    # Calculate probabilities and make cross-section plots
    probs_and_plots(wind_spd_cube_list, 'wind', start_vdt, end_vdt,
                    rec_m_date)
    probs_and_plots(temp_cube_list, 'temp', start_vdt, end_vdt, rec_m_date)
    probs_and_plots(rel_hum_cube_list, 'relative_humidity', start_vdt,
                    end_vdt, rec_m_date)
    rain_plots(rain_cube_list, start_vdt, end_vdt, rec_m_date)
    vis_plots(vis_cube_list, start_vdt, end_vdt, rec_m_date)

    # Update HTML page
    date_str = rec_m_date.strftime('%Y%m%d%HZ')
    update_html(date_str)



def housekeep(rec_m_date):
    """
    Removes MOGREPS-UK files older than earliest file needed.
    """
    # Date of oldest file currently needed
    first_dt = rec_m_date - timedelta(hours=5)

    # Loop through files in SCRATCH directory
    files = glob.glob(f'{SCRATCH_DIR}/*')
    for file in files:

        # Get filename
        fname = os.path.basename(file)

        # Ignore pickle file
        if 'pickle' in fname:
            continue

        # Get date/time of file
        f_year = int(fname[:4])
        f_month = int(fname[4:6])
        f_day = int(fname[6:8])
        f_hr = int(fname[9:11])

        # Filename datetime
        fdt = datetime(f_year, f_month, f_day, f_hr)

        # Remove file if necessary
        if fdt < first_dt:
            os.system(f'rm {file}')


def get_cubes(rec_m_date, start_vdt, end_vdt):

    # Start these as False and update later (only need to assign once)
    rot_lon, rot_lat, orog_cube = False, False, False

    # To add cubes to
    wind_spd_cube_list = iris.cube.CubeList([])
    temp_cube_list = iris.cube.CubeList([])
    rel_hum_cube_list = iris.cube.CubeList([])
    rain_cube_list = iris.cube.CubeList([])
    vis_cube_list = iris.cube.CubeList([])

    # Use multiprocessing to process each hour in parellel
    queue = Queue()
    processes = []

    # Get last 6 hours of MOGREPS-UK files (3 members per file)
    for hour in range(5, -1, -1):

        # Add to processes list for multiprocessing, using data_from_files
        args = (data_from_files, [start_vdt, end_vdt, rot_lat, rot_lon,
                                  orog_cube, hour, rec_m_date],
                queue)
        processes.append(Process(target=_mp_queue, args=args))

    # Start processes
    for process in processes:
        process.start()

    # Collect output from processes and close queue
    out_list = [queue.get() for _ in processes]
    queue.close()

    # Wait for all processes to complete before continuing
    for process in processes:
        process.join

    # Append output to cubelists
    for item in out_list:
        (wind_cubes, temp_cubes,
         rel_hum_cubes, rain_cubes, vis_cubes) = item
        for wind_cube in wind_cubes:
            wind_spd_cube_list.append(wind_cube)
        for temp_cube in temp_cubes:
            temp_cube_list.append(temp_cube)
        for rel_hum_cube in rel_hum_cubes:
            rel_hum_cube_list.append(rel_hum_cube)
        for rain_cube in rain_cubes:
            rain_cube_list.append(rain_cube)
        for vis_cube in vis_cubes:
            vis_cube_list.append(vis_cube)

    # Pickle data for later use if needed (to save time)
    uf.pickle_data([wind_spd_cube_list, temp_cube_list,
                    rel_hum_cube_list, rain_cube_list, vis_cube_list],
                   f'{SCRATCH_DIR}/pickle')

    return (wind_spd_cube_list, temp_cube_list, rel_hum_cube_list,
            rain_cube_list, vis_cube_list)


def get_vrbs():
    """
    Defines some variables required for extraction of data based on forecast
    period.
    """
    # Calculate period of forecast from first and last dts
    fcast_period = int((LAST_DT - FIRST_DT).total_seconds() / 3600) - 1

    # Time now (only to hour)
    now_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    # Issue date/time of most recent MOGREPS-UK file to use
    rec_m_date = now_hour - timedelta(hours=3)

    # Determine how far out to go based on the oldest MOGREPS-UK file used
    latest_lead_vdt = now_hour - timedelta(hours=8) + timedelta(hours=126)

    # Go as far out as possible up to day of forecast
    if latest_lead_vdt <= LAST_DT:
        start_vdt = latest_lead_vdt - timedelta(hours=fcast_period)
        end_vdt = latest_lead_vdt
    else:
        end_vdt = LAST_DT
        if rec_m_date >= FIRST_DT:
            start_vdt = rec_m_date
        else:
            start_vdt = FIRST_DT

    return rec_m_date, start_vdt, end_vdt


def convert_lat_lon(fname, lat, lon):
    """
    Converts standard lat/lon coordinates to rotated pole coordinates.
    """
    # Load cube
    cube = iris.load_cube(fname, U_CON)

    # Get rotated pole values
    pole_lon = cube.coord_system().grid_north_pole_longitude
    pole_lat = cube.coord_system().grid_north_pole_latitude

    # Convert latitude and longitude points
    rot_lon, rot_lat = rotate_pole(np.array(lon), np.array(lat), pole_lon,
                                   pole_lat)

    return rot_lat[0], rot_lon[0]


def surf_to_levels(cube_s, cube_m):
    """
    Changes coordinates and dimensions of surface cube to enable
    concatenating with model levels cube.
    """
    # Add dimension using height coordinate
    cube_s = iris.util.new_axis(cube_s, 'height')

    # Transpose to make cube of same shape as model levels cube
    cube_s.transpose([1, 0, 2, 3])

    # Use model levels cube as template for surface cube data
    cube_s = cube_m[:, 0:1, :, :].copy(data=cube_s.data)

    # Change coordinate values
    cube_s.coord('level_height').points = 1.5
    cube_s.coord('sigma').points = 1.0
    cube_s.coord('model_level_number').points = 0

    return cube_s


def get_wind_spd(fname, orog_cube, lat, lon, start_vdt, end_vdt):
    """
    Gets wind speed cube from U and V wind component cubes.
    """
    # Load U and V wind component cubes
    u_cube = iris.load_cube(fname, U_CON)
    v_cube = iris.load_cube(fname, V_CON)

    # Sample points for interpolating for site location lats/lons
    sample_pnts = [('grid_latitude', lat), ('grid_longitude', lon)]

    # Update cubes
    u_cube = update_cube(u_cube, sample_pnts, orog_cube, start_vdt, end_vdt,
                         'knots')
    v_cube = update_cube(v_cube, sample_pnts, orog_cube, start_vdt, end_vdt,
                         'knots')

    # Get wind speed cube
    wind_spd = (u_cube.data ** 2 + v_cube.data ** 2) ** 0.5
    wind_spd_cube = u_cube.copy(data=wind_spd)
    wind_spd_cube.standard_name = 'wind_speed'

    return wind_spd_cube


def get_temps(fname_s, fname_m, orog_cube, lat, lon, start_vdt, end_vdt):
    """
    Gets temperature cube, concatenating surface and model level cubes.
    """
    # Load cubes
    temp_cube_s = iris.load(fname_s, TEMP_1P5_CON)[0]
    temp_cube_m = iris.load_cube(fname_m, TEMP_CON)

    # Changes to enable concatenating
    temp_cube_s = surf_to_levels(temp_cube_s, temp_cube_m)

    # Concatenate surface and model levels cubes
    temp_cube = iris.cube.CubeList([temp_cube_s,
                                    temp_cube_m]).concatenate_cube()

    # Sample points for interpolating for site location lats/lons
    sample_pnts = [('grid_latitude', lat), ('grid_longitude', lon)]

    # Update cubes
    temp_cube = update_cube(temp_cube, sample_pnts, orog_cube, start_vdt,
                            end_vdt, 'celsius')

    return temp_cube


def get_rel_hums(fname_s, fname_m, orog_cube, lat, lon, start_vdt, end_vdt):
    """
    Gets relative humidity cube, calculating from air temperature, pressure
    and specific humidity for model levels, then concatenating with surface
    cube.
    """
    # Load cubes
    spec_hum_cube_m = iris.load_cube(fname_m, SPEC_HUM_CON)
    temp_cube_m = iris.load_cube(fname_m, TEMP_CON)
    pres_cube_m = iris.load_cube(fname_m, PRES_CON)
    rel_hum_cube_s = iris.load_cube(fname_s, REL_HUM_CON)

    # Changes to enable concatenating
    rel_hum_cube_s = surf_to_levels(rel_hum_cube_s, spec_hum_cube_m)
    rel_hum_cube_s.standard_name = 'relative_humidity'

    # Sample points for interpolating for site location lats/lons
    sample_pnts = [('grid_latitude', lat), ('grid_longitude', lon)]

    # Update cubes
    spec_hum_cube_m = update_cube(spec_hum_cube_m, sample_pnts, orog_cube,
                                  start_vdt, end_vdt, '')
    temp_cube_m = update_cube(temp_cube_m, sample_pnts, orog_cube, start_vdt,
                              end_vdt, 'celsius')
    pres_cube_m = update_cube(pres_cube_m, sample_pnts, orog_cube, start_vdt,
                              end_vdt, 'hPa')
    rel_hum_cube_s = update_cube(rel_hum_cube_s, sample_pnts, orog_cube,
                                 start_vdt, end_vdt, '')

    # Get pressure cube on same model levels as temp and humidity cubes
    sample_pnts = [('model_level_number',
                  temp_cube_m.coord('model_level_number').points)]
    pres_cube_m = pres_cube_m.interpolate(sample_pnts, iris.analysis.Linear())
    pres_cube_m = temp_cube_m.copy(data=pres_cube_m.data)

    # Convert model level cube from specific humidity to relative humidity
    rel_hum_cube_m = spec_hum_to_rel_hum(spec_hum_cube_m, pres_cube_m,
                                         temp_cube_m)

    # Concatenate surface and model levels cubes
    rel_hum_cube = iris.cube.CubeList([rel_hum_cube_s,
                                       rel_hum_cube_m]).concatenate_cube()

    # Add derived altitude coordinate back in (disappears after concatenating)
    fact = iris.aux_factory.HybridHeightFactory(
        rel_hum_cube.coord('level_height'),
        rel_hum_cube.coord('sigma'),
        rel_hum_cube.coord('surface_altitude'))

    rel_hum_cube.add_aux_factory(fact)

    return rel_hum_cube


def get_rains(fname, orog_cube, lat, lon, start_vdt, end_vdt):


    # Load cube
    cube = iris.load(fname, RAIN_CON)[0]

    # Sample points for interpolating for site location lats/lons
    sample_pnts = [('grid_latitude', lat), ('grid_longitude', lon)]

    # Interpolate horizontally using site lat/lon points
    cube = cube.interpolate(sample_pnts, iris.analysis.Linear())

    # Only use forecast valid for day of forecast
    cube_list = iris.cube.CubeList([])
    for ind, time_int in enumerate(cube.coord('time').points):
        vdt = cube.coord('time').units.num2date(time_int)
        if start_vdt <= vdt <= end_vdt:
            cube_list.append(cube[ind])

    # Merge into single cube
    new_cube = cube_list.merge_cube()

    # Add in realisation coordinate for control member to enable merging later
    # (use arbitrary value of 100)
    try:
        new_cube.coord('realization')
    except:
        real_coord = iris.coords.DimCoord(100, 'realization', units='1')
        new_cube.add_aux_coord(real_coord)

    # Convert units to mm hr-1
    new_cube *= 3600
    new_cube.units = 'mm hr-1'

    return new_cube


def get_vis(fname, orog_cube, lat, lon, start_vdt, end_vdt):


    # Load cube
    cube = iris.load(fname, VIS_CON)[0]

    # Sample points for interpolating for site location lats/lons
    sample_pnts = [('grid_latitude', lat), ('grid_longitude', lon)]

    # Interpolate horizontally using site lat/lon points
    cube = cube.interpolate(sample_pnts, iris.analysis.Linear())

    # Only use forecast valid for day of forecast
    cube_list = iris.cube.CubeList([])
    for ind, time_int in enumerate(cube.coord('time').points):
        vdt = cube.coord('time').units.num2date(time_int)
        if start_vdt <= vdt <= end_vdt:
            cube_list.append(cube[ind])

    # Merge into single cube
    new_cube = cube_list.merge_cube()

    # Add in realisation coordinate for control member to enable merging later
    # (use arbitrary value of 100)
    try:
        new_cube.coord('realization')
    except:
        real_coord = iris.coords.DimCoord(100, 'realization', units='1')
        new_cube.add_aux_coord(real_coord)

    return new_cube


def update_cube(cube, sample_pnts, orog_cube, start_vdt, end_vdt, c_units):
    """
    Gets smaller cube, interpolating using site lat/lon, removes forecasts not
    valid on forecast day, changes units appropriately and adds derived
    altitude coordinate.
    """
    # Interpolate horizontally using site lat/lon points
    cube = cube.interpolate(sample_pnts, iris.analysis.Linear())

    # Add orography as auxiliary coordinate to cube
    orog_coord = iris.coords.AuxCoord(orog_cube.data, 'surface_altitude',
                                      units='m')
    cube.add_aux_coord(orog_coord)

    # Define hybrid height coordinate
    fact = iris.aux_factory.HybridHeightFactory(cube.coord('level_height'),
                                                cube.coord('sigma'),
                                                cube.coord('surface_altitude'))

    # Add hybrid height coordinate to cube
    cube.add_aux_factory(fact)

    # Convert units if necessary (knots for wind, celsius for temps)
    if c_units:
        cube.convert_units(c_units)


    # Convert altitude units from metres to feet
    cube.coord('altitude').convert_units('feet')
    cube.coord('level_height').convert_units('feet')
    cube.coord('surface_altitude').convert_units('feet')

    # Only use forecast valid for day of forecast
    cube_list = iris.cube.CubeList([])
    for ind, time_int in enumerate(cube.coord('time').points):
        vdt = cube.coord('time').units.num2date(time_int)
        if start_vdt <= vdt <= end_vdt:
            cube_list.append(cube[ind])

    # Merge into single cube
    new_cube = cube_list.merge_cube()

    # Add in realisation coordinate for control member to enable merging later
    # (use arbitrary value of 100)
    try:
        new_cube.coord('realization')
    except:
        real_coord = iris.coords.DimCoord(100, 'realization', units='1')
        new_cube.add_aux_coord(real_coord)

    return new_cube


def calc_probs(cube, threshold, temp_thresh):
    """
    Returns cube with probabilities of wind speeds >= (or <= for temp 0C)
    threshold.
    """
    # Calculate probabilities of exceeding (or  under for temp 0C) threshold
    if threshold == 0 and temp_thresh:
        events = [(mem_cube.data <= threshold).astype(int)
                   for mem_cube in cube.slices_over("realization")]
    else:
        events = [(mem_cube.data >= threshold).astype(int)
                   for mem_cube in cube.slices_over("realization")]
    probs = (sum(events) / len(events)) * 100

    # Create cube using probs data
    probs_cube = cube[0].copy(data=probs)

    # Add altitude above ground level
    above_ground = (probs_cube.coord('altitude').points -
                    probs_cube.coord('surface_altitude').points)
    ground_coord = iris.coords.AuxCoord(above_ground.data,
                                        long_name='above_ground',
                                        units='ft')
    probs_cube.add_aux_coord(ground_coord, [0])

    return probs_cube


def x_plot(cube, issue_dt, param, threshold, temp_thresh, units):
    """
    Makes cross section plot over time.
    """
    # Colours and probability levels used in plot
    levels = [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 100]
    colors = ['#ffffff', '#e6ffe6', '#ccffcc', '#b3ffb3', '#99ff99', '#80ff80',
              '#80bfff', '#4da6ff', '#1a8cff', '#0073e6', '#0059b3', '#004080',
              '#004080']

    # For labelling
    param_str = param.replace('_', ' ')

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))

    # Get cross-section from cube
    cross_section = next(cube.slices(['time', 'model_level_number']))

    # Draw plot, using above_ground coordinate on y-axis
    contours = iplt.contourf(cross_section, levels=levels, colors=colors,
                             coords=['time', 'above_ground'])

    # Get rid of contour lines
    for contour in contours.collections:
        contour.set_edgecolor("face")

    # Set limits and labels
    ax.set_ylim(0, 200)
    ax.set_ylabel('Altitude above ground level (ft)')

    # Format dates
    plt.gca().xaxis.axis_date()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y\n%HZ'))

    # Define parameter specific labels
    if temp_thresh:
        if threshold == 0:
            label_type = 'less than'
        else:
            label_type = 'exceeding'
    else:
        label_type = 'exceeding'

    # Add colour bar
    plt.subplots_adjust(bottom=0.23)
    cbaxes_probs = fig.add_axes([0.12, 0.1, 0.78, 0.02])
    cbar_probs = plt.colorbar(contours, cax=cbaxes_probs,
                              orientation='horizontal')
    cbar_probs.set_ticks(levels)
    cbar_probs.set_ticklabels(['{}%'.format(perc) for perc in levels])
    cbar_probs.ax.tick_params(labelsize=10)
    cbar_probs.set_label(f'Probability of {param_str} {label_type} '
                         f'{threshold} {units}', fontsize=12)

    # Figure title
    fig.suptitle(f'MOGREPS-UK {param_str} probabilities - cross-section over '
                 'time', fontsize=25)

    # Save figure and close plot
    date_str = issue_dt.strftime('%Y%m%d%HZ')
    fname = (f'{HTML_DIR}/images/{SITE_NAME.replace(" ", "_")}/'
             f'mogreps_x_section_{date_str}_{param}_{threshold}.png')
    fig.savefig(fname)
    plt.close()


def rain_probs(cube, threshold):

    # Calculate probabilities of exceeding threshold
    events = [(mem_cube.data >= threshold).astype(int)
               for mem_cube in cube.slices_over("realization")]
    probs = (sum(events) / len(events)) * 100

    # Create cube using probs data
    probs_cube = cube[0].copy(data=probs)

    return probs_cube


def vis_probs(cube, threshold):

    # Calculate probabilities of exceeding threshold
    events = [(mem_cube.data <= threshold).astype(int)
               for mem_cube in cube.slices_over("realization")]
    probs = (sum(events) / len(events)) * 100

    # Create cube using probs data
    probs_cube = cube[0].copy(data=probs)

    return probs_cube


def rain_plots(cube_list, start_vdt, end_vdt, m_date):
    """
    Makes cross section plot over time.
    """
    # Number of 5 minute forecast periods
    num_fps = int((end_vdt - start_vdt).total_seconds() / 300)

    # Make empty cube list to append to for each threshold
    prob_lists = [iris.cube.CubeList([]) for _ in RAIN_THRESHS]

    # List of valid datetimes
    vdts = list(rrule(MINUTELY, dtstart=start_vdt, interval=5, count=num_fps))

    # Dictionary with empty cubelist assigned to each 5 minute valid datetime
    five_min_cubes = {vdt: iris.cube.CubeList([]) for vdt in vdts}

    # Loop through each cube
    for cube in cube_list:
        # Remove forecast period coordinate to allow merging
        cube.remove_coord('forecast_period')
        # For each cube time append to appropriate cubelist in five_min_cubes
        for ind, time in enumerate(cube.coord('time').points):
            cube_dt = cube.coord('time').units.num2date(time)
            if cube_dt in vdts:
                if len(cube.coord('time').points) == 1:
                    five_min_cubes[cube_dt].append(cube)
                else:
                    five_min_cubes[cube_dt].append(cube[ind])

    hr_vdts = set(vdt.replace(minute=0) for vdt in vdts)
    hour_cubes = {vdt: iris.cube.CubeList([]) for vdt in hr_vdts}

    # Merge cubelists into single cubes
    for vdt in five_min_cubes:
        merged_five_min_cube = five_min_cubes[vdt].merge(unique=False)[0]
        hour_cubes[vdt.replace(minute=0)].append(merged_five_min_cube)

    # Convert to probabilities for each threshold for each hour cube
    for vdt in hour_cubes:

        # Merge cube
        hour_cube = hour_cubes[vdt].merge_cube()

        # Collapse cube taking max rate in hour
        max_in_hour_cube = hour_cube.collapsed('time', iris.analysis.MAX)

        prob_cubes = [rain_probs(max_in_hour_cube, thresh)
                      for thresh in RAIN_THRESHS]

        # Append probability cubes to cube lists
        [prob_list.append(prob_cube)
         for prob_list, prob_cube in zip(prob_lists, prob_cubes)]

    # Merge cubes
    merged_probs = [prob_list.merge_cube() for prob_list in prob_lists]

    # Make some plots
    [prob_plot(probs_cube, m_date, thresh, 'rain',
               'max rate rate in hour exceeding', 'mm hr-1')
     for probs_cube, thresh in zip(merged_probs, RAIN_THRESHS)]


def vis_plots(cube_list, start_vdt, end_vdt, m_date):
    """
    Makes cross section plot over time.
    """
    # Make empty cube list to append to for each threshold
    prob_lists = [iris.cube.CubeList([]) for _ in VIS_THRESHS]

    # Number of 1 hour forecast periods
    num_fps = int((end_vdt - start_vdt).total_seconds() / 3600)

    # List of hourly dts in forecast period
    vdts = rrule(HOURLY, dtstart=start_vdt, interval=1, count=num_fps)
    hour_cubes = {vdt: iris.cube.CubeList([]) for vdt in vdts}

    # Get all forecasts valid for each hour in forecast period
    for cube in cube_list:

        # For each cube time append to appropriate cubelist in five_min_cubes
        for ind, time in enumerate(cube.coord('time').points):
            cube_dt = cube.coord('time').units.num2date(time)
            if cube_dt in vdts:
                if len(cube.coord('time').points) == 1:
                    hour_cubes[cube_dt].append(cube)
                else:
                    hour_cubes[cube_dt].append(cube[ind])

    # Convert to probabilities for each threshold for each hour cube
    for vdt in hour_cubes:

        # Merge cube
        hour_cube = hour_cubes[vdt].merge_cube()

        prob_cubes = [vis_probs(hour_cube, thresh) for thresh in VIS_THRESHS]

        # Append probability cubes to cube lists
        [prob_list.append(prob_cube)
         for prob_list, prob_cube in zip(prob_lists, prob_cubes)]

    # Merge cubes
    merged_probs = [prob_list.merge_cube() for prob_list in prob_lists]

    # Make some plots
    [prob_plot(probs_cube, m_date, thresh, 'vis', '1.5m visibility below', 'm')
     for probs_cube, thresh in zip(merged_probs, VIS_THRESHS)]


def prob_plot(cube, issue_dt, threshold, wx_type, title_str, units):

    # Colours and probability levels used in plot
    levels = np.array([0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 100])
    colors = np.array(['#ffffff', '#e6ffe6', '#ccffcc', '#b3ffb3', '#99ff99',
                       '#80ff80', '#80bfff', '#4da6ff', '#1a8cff', '#0073e6',
                       '#0059b3', '#004080', '#004080'])

    # Set colours of scatter plot markers
    prob_colors = []
    for prob in cube.data:
        prob_colors.append(colors[levels <= prob][-1])

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))

    tcoord = cube.coord('time')
    dates = [tcoord.units.num2date(point).strftime('%d-%m-%Y\n%HZ')
             for point in tcoord.points]

    # Gap between x ticks, depending on number of days shown on plot
    gap = int(len(dates) / 8)

    # Define x axis ticks and labels
    xtick_locs, xlabels = [], []
    for ind, date in enumerate(dates):
        if gap == 0 or ind % gap == 0 or date == dates[-1]:
            xtick_locs.append(ind)
            xlabels.append (date)

    # Make colour coded scatter plot
    ax.bar(dates, cube.data, width=1.0, color=prob_colors)

    # Set limits, labels and legend
    ax.set_ylim(0, 100)
    ax.set_ylabel('Probability (%)', fontsize=17)
    ax.set_xticks(xtick_locs)
    ax.set_xticklabels(xlabels, fontsize=12)

    # Figure title
    fig.suptitle(f'MOGREPS-UK probabilities of {title_str} {threshold}{units}',
                 fontsize=22)

    # Save figure and close plot
    date_str = issue_dt.strftime('%Y%m%d%HZ')
    fname = (f'{HTML_DIR}/images/{SITE_NAME.replace(" ", "_")}/'
             f'mogreps_x_section_{date_str}_{wx_type}_{threshold}.png')
    fig.savefig(fname)
    plt.close()


def update_html(date):
    """
    Updates html file.
    """
    # File name of html file
    html_fname = (f'{HTML_DIR}/html/{SITE_NAME.replace(" ", "_")}'
                  '_mog_uk_fcasts.shtml')

    # Make new directories/files if needed
    if not os.path.exists(html_fname):

        # Make html file starting with template
        template = f'{HTML_DIR}/html/mog_template.shtml'
        os.system(f'cp {template} {html_fname}')

        # Put in trial-specific stuff
        # Get lines from template
        file = open(html_fname, 'r')
        lines = file.readlines()
        file.close()

        # Change bits specific to trial
        site_fname = SITE_NAME.replace(' ', '_')
        lines[34] = lines[34].replace('TRIAL', site_fname)
        lines[48] = lines[48].replace('NAME', SITE_NAME)
        lines[48] = lines[48].replace('HEIGHT', str(SITE_HEIGHT))
        lines[76] = lines[76].replace('DATE', date)
        lines[79] = lines[79].replace('TRIAL', site_fname)
        lines[79] = lines[79].replace('NAME', SITE_NAME)
        lines[88] = lines[88].replace('TRIAL', site_fname)
        lines[88] = lines[88].replace('DATE', date)

        # Assign to new_lines
        new_lines = lines

        # Add site to sidebar
        file = open(SIDEBAR, 'r')
        lines = file.readlines()
        file.close()

        # Split up lines and append new line
        first_lines = lines[:-5]
        last_lines = lines[-5:]
        url = f'{URL_START}/{site_fname}_mog_uk_fcasts.shtml'
        first_lines.append(f'          <li><a href="{url}">'
                           f'{SITE_NAME} MOGREPS-UK Forecasts</a></li>\n')

        # Concatenate the lists together and re-write the lines to a new file
        side_lines = first_lines + last_lines
        file = open(SIDEBAR, 'w')
        for line in side_lines:
            file.write(line)
        file.close()

    else:
        # Read in existing file, getting 2 lists of lines from the file, split
        # where an extra line is required
        file = open(html_fname, 'r')
        lines = file.readlines()
        file.close()
        first_lines = lines[:-19]
        last_lines = lines[-19:]

        # Edit html file and append/edit the required lines
        first_lines[-1] = first_lines[-1].replace(' selected="selected"', '')
        first_lines.append('                        <option selected='
                           f'"selected" value="{date}">{date}</option>\n')
        last_lines[-8] = last_lines[-8].replace(last_lines[-8][-82:-71], date)

        # Concatenate the lists together
        new_lines = first_lines + last_lines

    # Re-write the lines to a new file
    file = open(html_fname, 'w')
    for line in new_lines:
        file.write(line)
    file.close()


def get_fname_strs(m_date, start_vdt, end_vdt):
    """
    Determines member numbers and lead times to use.
    """
    # Define date string for filenames
    date_str = m_date.strftime('%Y%m%dT%H00Z')

    # Read filenames on HPC, temporarily ssh-ing onto HPC
    dirs = subprocess.Popen([f'ssh -Y xcel01 ls {HPC_DIR}/{date_str}/'],
                            stdout=subprocess.PIPE, shell=True)
    (out, _) = dirs.communicate()

    # Get member numbers from filenames
    member_strs = []
    for string in str(out).split('enuk_um_'):
        if string[:3].isnumeric():
            member_strs.append(string[:3])

    # Determine lead time numbers used for filenames required, based on day of
    # forecast and MOGREPS-UK run
    f_nums = []
    for num in FNAME_NUMS:

        # Append to f_nums if file contains data for forecast date
        for lead in range(int(num), int(num) + 4):

            # Get valid date
            vdt = m_date + timedelta(hours=lead)

            # Check if file has any relevant valid dates in it and append to
            # list if so
            if start_vdt <= vdt <= end_vdt:
                f_nums.append(num)
                break

    return member_strs, f_nums


def lat_lon_orog(m_date, member_str, hour):
    """
    Converts standard lat/lon to rotated pole coordinates and gets orography
    cube interpolated to rotated pole lat/lon.
    """
    # Define date string for filenames
    date_str = m_date.strftime('%Y%m%dT%H00Z')

    # Define filename on HPC and target filename on scratch
    fpath = (f'{USER}@xcel01:{HPC_DIR}/{date_str}/enuk_um_{member_str}/'
             'enukaa_pd000')
    scratch_fname = (f'{SCRATCH_DIR}/{date_str}_enukaa_pd000_{hour}_'
                     f'{member_str}')

    # Copy file from HPC to scratch directory
    if not os.path.exists(scratch_fname):
        os.system(f'scp {fpath} {scratch_fname}')

    # If successful, get rotated lat/lon and orography cube from file
    if os.path.exists(scratch_fname):

        # Convert lat/lon to rotated pole coordinates
        rot_lat, rot_lon = convert_lat_lon(scratch_fname, LAT, LON)

        # Get orography cube
        orog_cube = iris.load_cube(scratch_fname, OROG_CON)

        # Linearly interpolate orography cube for site location lats/lons
        sample_pnts = [('grid_latitude', rot_lat), ('grid_longitude', rot_lat)]
        orog_cube = orog_cube.interpolate(sample_pnts, iris.analysis.Linear())

    # Otherwise keep variables as False
    else:
        rot_lat, rot_lon, orog_cube = False, False, False

    return rot_lat, rot_lon, orog_cube


def copy_from_hpc(f_num, m_date, member_str, hour):
    """
    Copies file from HPC to scratch directory.
    """
    # Define date string for filenames
    date_str = m_date.strftime('%Y%m%dT%H00Z')

    # List to append filenames to
    scratch_fnames = []

    # enukaa_pd*** and enukaa_pe*** files needed
    for letter in ['d', 'e']:

        # Define filenames
        fname = 'enukaa_p{}{}'.format(letter, f_num)
        fpath = (f'{USER}@xcel01:{HPC_DIR}/{date_str}/enuk_um_{member_str}/'
                 f'{fname}')
        scratch_fname = f'{SCRATCH_DIR}/{date_str}_{fname}_{hour}_{member_str}'

        # Copy to scratch directory (if necessary)
        if not os.path.exists(scratch_fname):
            os.system(f'scp {fpath} {scratch_fname}')

        # Append scratch filename to list
        scratch_fnames.append(scratch_fname)

    return scratch_fnames


def probs_and_plots(cube_list, param, start_vdt, end_vdt, m_date):
    """
    Calculates probabilities and makes cross-section plots.
    """
    # Define parameter-specific variables
    if param == 'wind':
        thresholds = WIND_THRESHS
        units = 'knots'
        temp_thresh = False
    elif param == 'relative_humidity':
        thresholds = REL_HUM_THRESHS
        units = '%'
        temp_thresh = False
    else:
        thresholds = TEMP_THRESHS
        units = 'Celsius'
        temp_thresh = True

    # Make empty cube list to append to for each threshold
    prob_lists = [iris.cube.CubeList([]) for _ in thresholds]

    # Number of 1 hour forecast periods
    num_fps = int((end_vdt - start_vdt).total_seconds() / 3600)

    # Get all forecasts valid for each hour in forecast period
    for vdt in rrule(HOURLY, dtstart=start_vdt, interval=1, count=num_fps):

        # Cube list to append to
        hour_cube_list = iris.cube.CubeList([])

        # Find forecasts valid for hour and append to cube list
        for cube in cube_list:
            for time_cube in cube.slices_over("time"):
                time_int = time_cube.coord('time').points[0]
                if cube.coord('time').units.num2date(time_int) == vdt:
                    hour_cube_list.append(time_cube)
        # Merge into single cube
        hour_cubes = hour_cube_list.merge(unique=False)
        hour_cube = hour_cubes[0]

        # Convert to probabilities for each threshold
        prob_cubes = [calc_probs(hour_cube, thresh, temp_thresh)
                      for thresh in thresholds]

        # Append probability cubes to cube lists
        [prob_list.append(prob_cube)
         for prob_list, prob_cube in zip(prob_lists, prob_cubes)]

    # Merge cubes
    merged_probs = [prob_list.merge_cube() for prob_list in prob_lists]

    # Make cross section plots
    [x_plot(probs_cube, m_date, param, thresh, temp_thresh, units)
     for probs_cube, thresh in zip(merged_probs, thresholds)]


def data_from_files(start_vdt, end_vdt, rot_lat, rot_lon, orog_cube, hour,
                    rec_m_date):
    """
    Gets data from MOGREPS-UK files, if possible, then sorts out data and
    returns lists of cubes.
    """
    # Issue date/time of appropriate MOGREPS-UK file
    m_date = rec_m_date - timedelta(hours=hour)

    # To append cubes to
    (wind_cubes, temp_cubes,
     rel_hum_cubes, rain_cubes, vis_cubes) = [], [], [], [], []

    # Determine ensemble member numbers used and lead times to use
    member_strs, f_nums = get_fname_strs(m_date, start_vdt, end_vdt)

    # If none found, print message
    if not member_strs:
        print('member_strs MISSING')

    # Copy files accross from HPC and get data from them
    for member_str in member_strs:

        # Convert lat/lon and get constraints and get orography cube
        # (if needed - only need to do this once)
        if not rot_lat or not rot_lon or not orog_cube:
            rot_lat, rot_lon, orog_cube = lat_lon_orog(m_date, member_str,
                                                       hour)

        # Load in each relevant file and get cubes
        for f_num in f_nums:

            # Copy surface and model level files across from HPC
            scratch_s, scratch_m = copy_from_hpc(f_num, m_date, member_str,
                                                 hour)

            # Only continue if files have successfully been copied across
            if (os.path.exists(scratch_s) and os.path.exists(scratch_m) and
                orog_cube):

                # Get wind speed cube
                try:
                    wind_spd = get_wind_spd(scratch_m, orog_cube, rot_lat,
                                            rot_lon, start_vdt, end_vdt)
                    wind_cubes.append(wind_spd)
                except:
                    print('Winds failed')

                # Get temperature cube
                try:
                    temps = get_temps(scratch_s, scratch_m, orog_cube, rot_lat,
                                      rot_lon, start_vdt, end_vdt)
                    temp_cubes.append(temps)
                except:
                    print('Temps failed')

                # Get relative humidity cube
                try:
                    rel_hums = get_rel_hums(scratch_s, scratch_m, orog_cube,
                                            rot_lat, rot_lon, start_vdt,
                                            end_vdt)
                    rel_hum_cubes.append(rel_hums)
                except:
                    print('Humidity failed')

                # Get precip cube
                try:
                    rains = get_rains(scratch_s, orog_cube, rot_lat, rot_lon,
                                      start_vdt, end_vdt + timedelta(hours=1))
                    rain_cubes.append(rains)
                except:
                    print('Rain failed')

                # Get visibility cube
                try:
                    vis = get_vis(scratch_s, orog_cube, rot_lat, rot_lon,
                                  start_vdt, end_vdt)
                    vis_cubes.append(vis)
                except:
                    print('Vis failed')

            # Otherwise, print message
            else:
                print('FILE(S) MISSING')

    return wind_cubes, temp_cubes, rel_hum_cubes, rain_cubes, vis_cubes


def spec_hum_to_rel_hum(spec_hum_cube, pressure_cube, t_dry_cube):
    """
    Converts specific humidity to relative humidity.
    """
    # Get data from cubes
    spec_hum, pressure, t_dry = [cube.data for cube in
                                 [spec_hum_cube, pressure_cube, t_dry_cube]]

    # Calculate vapour pressure
    vap_pres = spec_hum * pressure / (REPSILON + (1 - REPSILON) * spec_hum)

    # Calculate saturated vapour pressure (Magnus formula (WMO CIMO 2008)
    corr = 1.0016 + 3.15 * 10 ** (-6.0) * pressure - (0.074 / pressure)
    sat_vap_pres = corr * 6.112 * np.exp(17.62 * t_dry / (t_dry + 243.12))

    # Calculate relative humidity
    rel_hum = 100.0 * vap_pres / sat_vap_pres

    # Create new cube of relative humidities, using spec_hum_cube as template
    rel_hum_cube = spec_hum_cube.copy(data=rel_hum)
    rel_hum_cube.standard_name = 'relative_humidity'

    return rel_hum_cube


def _mp_queue(function, args, queue):

    """ Wrapper function for allowing multiprocessing of a function and
    ensuring that the output is appended to a queue, to be picked up later.
    """
    queue.put(function(*args))


if __name__ == "__main__":

    # Print time
    time_1 = uf.print_time('started')

    try:
        new_data = sys.argv[1]
    except:
        print('WARNING! Arguments not set correctly so will exit python '
              'script')
        exit()

    main(new_data)

    # Print time
    time_2 = uf.print_time('Finished')

    # Print time taken
    uf.time_taken(time_1, time_2, unit='minutes')
