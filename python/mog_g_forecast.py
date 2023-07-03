""" 
Script to create MOGREPS-G cross-section plots.
"""
import numpy as np
import os
import glob
import sys
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
    MOG_G_DIR = os.environ['MOG_G_DIR']
    SCRATCH_DIR = os.environ['SCRATCH_DIR']
    HTML_DIR = os.environ['HTML_DIR']
    URL_START = os.environ['URL_START']
    SIDEBAR = os.environ['SIDEBAR']
    MASS_DIR = os.environ['MASS_DIR']
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
FIRST_DT = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
LAST_DT = FIRST_DT + timedelta(hours=24)
# Location/height/name of site
LAT = 33.035
LON = -106.407
SITE_HEIGHT = 40
SITE_NAME = 'White Sands'
SITE_FNAME = SITE_NAME.replace(' ', '_')

# ==============================================================================

# Lead time numbers used in filenames
FNAME_NUMS = [str(num).zfill(3) for num in range(0, 201, 3)]
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


def get_wind_spd(fname, member):
    """
    Gets wind speed cube from U and V wind component cubes.
    """
    # Load U and V wind component cubes
    u_cube = iris.load_cube(fname, U_CON)
    v_cube = iris.load_cube(fname, V_CON)

    # Update cubes
    u_cube = update_cube(u_cube, member, 'knots')
    v_cube = update_cube(v_cube, member, 'knots')

    # Only continue if updating cube successful
    if not (u_cube and v_cube):
        return False

    # Get wind speed cube
    wind_spd = (u_cube.data ** 2 + v_cube.data ** 2) ** 0.5
    wind_spd_cube = u_cube.copy(data=wind_spd)
    wind_spd_cube.standard_name = 'wind_speed'

    return wind_spd_cube


def get_temps(fname_d, fname_f, member):
    """
    Gets temperature cube, concatenating surface and model level cubes.
    """
    # Load cubes
    temp_cube_s = iris.load(fname_d, TEMP_1P5_CON)[0]
    temp_cube_m = iris.load_cube(fname_f, TEMP_CON)

    # Changes to enable concatenating
    temp_cube_s = surf_to_levels(temp_cube_s, temp_cube_m)

    # Concatenate surface and model levels cubes
    temp_cube = iris.cube.CubeList([temp_cube_s,
                                    temp_cube_m]).concatenate_cube()

    # Update cubes
    temp_cube = update_cube(temp_cube, member, 'celsius')

    return temp_cube


def get_rel_hums(fname_d, fname_f, member):
    """
    Gets relative humidity cube, calculating from air temperature, pressure
    and specific humidity for model levels, then concatenating with surface
    cube.
    """
    # Load cubes
    spec_hum_cube_m = iris.load_cube(fname_f, SPEC_HUM_CON)
    temp_cube_m = iris.load_cube(fname_f, TEMP_CON)
    pres_cube_m = iris.load_cube(fname_f, PRES_CON)

    # Get rid of surface model level in specific humidity cube (so they all
    # have 70 levels)
    spec_hum_cube_m = spec_hum_cube_m[:, 1:, :, :]

    # Update cubes
    spec_hum_cube_m = update_cube(spec_hum_cube_m, member, '')
    temp_cube_m = update_cube(temp_cube_m, member, 'celsius')
    pres_cube_m = update_cube(pres_cube_m, member, 'hPa')

    # Only continue if updating cubes successful
    if not all([spec_hum_cube_m, temp_cube_m, pres_cube_m]):
        return False

    # Get pressure cube on same model levels as temp and humidity cubes
    sample_pnts = [('model_level_number',
                    temp_cube_m.coord('model_level_number').points)]
    pres_cube_m = pres_cube_m.interpolate(sample_pnts, iris.analysis.Linear())
    pres_cube_m = temp_cube_m.copy(data=pres_cube_m.data)

    # Convert model level cube from specific humidity to relative humidity
    rel_hum_cube = spec_hum_to_rel_hum(spec_hum_cube_m, pres_cube_m,
                                       temp_cube_m)

    return rel_hum_cube


def get_rains(fname, member):

    # Load cube
    cube = iris.load(fname, RAIN_CON)[0]

    # Sample points for interpolating for site location lats/lons
    sample_pnts = [('latitude', LAT), ('longitude', LON)]
    cube = cube.interpolate(sample_pnts, iris.analysis.Linear())

    # Only use forecast valid for day of forecast
    cube_list = iris.cube.CubeList([])
    for ind, time_int in enumerate(cube.coord('time').points):
        vdt = cube.coord('time').units.num2date(time_int)
        if FIRST_DT <= vdt <= LAST_DT + timedelta(hours=1):
            cube_list.append(cube[ind])

    # Merge into single cube (if possible)
    try:
        new_cube = cube_list.merge_cube()
    except:
        print('Empty CubeList (rain)')
        return False

    # Add in realisation coordinate
    try:
        new_cube.coord('realization')
    except:
        real_coord = iris.coords.DimCoord(member, 'realization', units='1')
        new_cube.add_aux_coord(real_coord)

    # Convert units to mm hr-1
    new_cube *= 3600
    new_cube.units = 'mm hr-1'

    return new_cube


def get_vis(fname, member):

    # Load cube
    cube = iris.load(fname, VIS_CON)[0]

    # Sample points for interpolating for site location lats/lons
    sample_pnts = [('latitude', LAT), ('longitude', LON)]
    cube = cube.interpolate(sample_pnts, iris.analysis.Linear())

    # Only use forecast valid for day of forecast
    cube_list = iris.cube.CubeList([])
    for ind, time_int in enumerate(cube.coord('time').points):
        vdt = cube.coord('time').units.num2date(time_int)
        if FIRST_DT <= vdt <= LAST_DT:
            cube_list.append(cube[ind])

    # Merge into single cube (if possible)
    try:
        new_cube = cube_list.merge_cube()
    except:
        print('Empty CubeList (rain)')
        return False

    # Add in realisation coordinate
    try: 
        new_cube.coord('realization')
    except:
        real_coord = iris.coords.DimCoord(member, 'realization', units='1')
        new_cube.add_aux_coord(real_coord)

    return new_cube


def update_cube(cube, member, c_units):
    """
    Gets smaller cube, interpolating using site lat/lon, removes forecasts not
    valid on forecast day, changes units appropriately and adds derived
    altitude coordinate.
    """
    # Sample points for interpolating for site location lats/lons
    sample_pnts = [('latitude', LAT), ('longitude', LON)]

    # Interpolate horizontally using site lat/lon points
    cube = cube.interpolate(sample_pnts, iris.analysis.Linear())

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
        if FIRST_DT <= vdt <= LAST_DT:
            cube_list.append(cube[ind])

    # Merge into single cube
    try:
        new_cube = cube_list.merge_cube()
    except:
        print('Empty Cubelist (update_cube line 282)')
        return False

    # Add in realisation coordinate if needed
    try:
        new_cube.coord('realization')
    except:
        real_coord = iris.coords.DimCoord(member, 'realization', units='1')
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


def x_plot(cube, g_hour, param, threshold, temp_thresh, units):
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
    fig.suptitle(f'MOGREPS-G {param_str} probabilities - cross-section over '
                 'time', fontsize=25)

    # Save figure and close plot
    date_str = g_hour.strftime('%Y%m%d%HZ')
    fname = (f'{HTML_DIR}/images/{SITE_FNAME}/mogreps_x_section_{date_str}'
             f'_{param}_{threshold}.png')
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


def rain_plots(cube_list, g_hour):
    """
    Makes cross section plot over time.
    """
    # Number of 5 minute forecast periods
    num_fps = int((LAST_DT - FIRST_DT).total_seconds() / 300)

    # Make empty cube list to append to for each threshold
    prob_lists = [iris.cube.CubeList([]) for _ in RAIN_THRESHS]

    # List of valid datetimes
    vdts = list(rrule(MINUTELY, dtstart=FIRST_DT, interval=5, count=num_fps))

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
    [prob_plot(probs_cube, g_hour, thresh, 'rain',
               'max rate rate in hour exceeding', 'mm hr-1')
     for probs_cube, thresh in zip(merged_probs, RAIN_THRESHS)]


def vis_plots(cube_list, g_hour):
    """
    Makes cross section plot over time.
    """
    # Make empty cube list to append to for each threshold
    prob_lists = [iris.cube.CubeList([]) for _ in VIS_THRESHS]

    # Number of 1 hour forecast periods
    num_fps = int((LAST_DT - FIRST_DT).total_seconds() / 3600)

    # List of hourly dts in forecast period
    vdts = rrule(HOURLY, dtstart=FIRST_DT, interval=1, count=num_fps)
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
    [prob_plot(probs_cube, g_hour, thresh, 'vis', '1.5m visibility below', 'm')
     for probs_cube, thresh in zip(merged_probs, VIS_THRESHS)]


def prob_plot(cube, g_hour, threshold, wx_type, title_str, units):

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
    fig.suptitle(f'MOGREPS-G probabilities of {title_str} {threshold}{units}',
                 fontsize=22)

    # Save figure and close plot
    date_str = g_hour.strftime('%Y%m%d%HZ')
    fname = (f'{HTML_DIR}/images/{SITE_FNAME}/mogreps_x_section_{date_str}'
             f'_{wx_type}_{threshold}.png')
    fig.savefig(fname)
    plt.close()


def update_html(date, site_height, site_name, site_fname):
    """
    Updates html file.
    """
    # File name of html file and images/MASS directories
    html_fname = f'{HTML_DIR}/html/{SITE_FNAME}_mog_uk_fcasts.shtml'
    img_dir = f'{HTML_DIR}/images/{SITE_FNAME}'
    mass_s_dir = f'{MASS_DIR}/{SITE_FNAME}'

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
        lines[34] = lines[34].replace('TRIAL', SITE_FNAME)
        lines[48] = lines[48].replace('NAME', SITE_NAME)
        lines[48] = lines[48].replace('HEIGHT', str(SITE_HEIGHT))
        lines[76] = lines[76].replace('DATE', date)
        lines[79] = lines[79].replace('TRIAL', SITE_FNAME)
        lines[79] = lines[79].replace('NAME', SITE_NAME)
        lines[88] = lines[88].replace('TRIAL', SITE_FNAME)
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
        url = f'{URL_START}/{SITE_FNAME}_mog_uk_fcasts.shtml'
        first_lines.append(f'          <li><a href="{url}">'
                           f'{SITE_NAME} MOGREPS-G Forecasts</a></li>\n')

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

        # Remove images if more than a week old
        for line in reversed(first_lines):
            
            # Stop if reached the start of the dropdowm menu
            if 'select id' in line:
                break

            # Otherwise, get date and remove if more than 1 week old
            if line[39:49].isnumeric():
                vdt = datetime(int(line[39:43]), int(line[43:45]), 
                               int(line[45:47]), int(line[47:49]))
                if (datetime.utcnow() - vdt).days >= 7:
                    first_lines.remove(line)    

                    # Also archive images
                    img_fnames = glob.glob(f'{img_dir}/*{line[39:49]}*')
                    for img_fname in img_fnames:
                        just_fname = os.path.basename(img_fname)
                        os.system(f'tar -zcvf {img_fname}.tar.gz {img_fname}')
                        os.system(f'moo put {img_fname}.tar.gz {mass_s_dir}')
                        os.system(f'rm {img_fname}.tar.gz {img_fname}')

        # Concatenate the lists together
        new_lines = first_lines + last_lines

    # Re-write the lines to a new file
    file = open(html_fname, 'w')
    for line in new_lines:
        file.write(line)
    file.close()


def get_orog(lat, lon, member, m_dir_path):
    """
    Gets orography cube interpolated to lat/lon values.
    """
    # Define filename on HPC and target filename on scratch
    fpath = (f'{m_dir_path}/englaa_pf000')
    scratch_fname = f'{SCRATCH_DIR}/{member}_englaa_pf000'

    # Copy file from HPC to scratch directory
    os.system(f'scp {fpath} {scratch_fname}')

    # If successful, get rotated lat/lon and orography cube from file
    if os.path.exists(scratch_fname):

        # Get orography cube
        orog_cube = iris.load_cube(scratch_fname, OROG_CON)

        # Linearly interpolate orography cube for site location lats/lons
        sample_pnts = [('latitude', lat), ('longitude', lon)]
        orog_cube = orog_cube.interpolate(sample_pnts, iris.analysis.Linear())

        # Remove file from scratch directory
        os.system(f'rm {scratch_fname}')

    # Otherwise keep variables as False
    else:
        lat, lon, orog_cube = False, False, False
    
    return orog_cube


def copy_from_hpc(f_num, latest_mog_dir, member):
    """
    Copies file from HPC to scratch directory.
    """
    # List to append filenames to
    scratch_fnames = []

    # enukaa_pd*** and enukaa_pf*** files needed
    for letter in ['d', 'f']:

        # Define filenames
        fname = 'englaa_p{}{}'.format(letter, f_num)
        fpath = (f'{latest_mog_dir}/{fname}')
        scratch_fname = f'{SCRATCH_DIR}/{member}_{fname}'

        # Copy to scratch directory and append scratch filename to list
        os.system(f'scp {fpath} {scratch_fname}')
        scratch_fnames.append(scratch_fname)

    return scratch_fnames


def probs_and_plots(cube_list, param, g_hour):
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
    num_fps = int((LAST_DT - FIRST_DT).total_seconds() / 3600)

    # Get all forecasts valid for each hour in forecast period
    for vdt in rrule(HOURLY, dtstart=FIRST_DT, interval=1, count=num_fps):

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
    [x_plot(probs_cube, g_hour, param, thresh, temp_thresh, units)
     for probs_cube, thresh in zip(merged_probs, thresholds)]


def data_from_files(latest_mog_dir, f_nums, member):
    """
    Gets data from MOGREPS-G files, if possible, then sorts out data and
    returns lists of cubes.
    """
    # Get member directory
    m_dir_name = f'engl_um_{member:03}'
    m_dir_path = f'{latest_mog_dir}/{m_dir_name}'

    # To append cubes to
    (wind_cubes, temp_cubes, 
     rel_hum_cubes, rain_cubes, vis_cubes) = [], [], [], [], []

    # Load in each relevant file and get cubes
    for f_num in f_nums:

        # Copy surface and model level files across from HPC
        scratch_d, scratch_f = copy_from_hpc(f_num, m_dir_path, member)

        # Only continue if files have successfully been copied across
        if not (os.path.exists(scratch_d) and os.path.exists(scratch_f)):
            print('FILE(S) MISSING')
            continue

        # Get wind speed cube and append to list if successful
        wind_spd = get_wind_spd(scratch_f, member)
        if wind_spd:
            wind_cubes.append(wind_spd)

        # Get temperature cube and append to list if successful
        temps = get_temps(scratch_d, scratch_f, member)
        if temps:
            temp_cubes.append(temps)

        # Get relative humidity cube and append to list if successful
        rel_hums = get_rel_hums(scratch_d, scratch_f, member)
        if rel_hums:
            rel_hum_cubes.append(rel_hums)

        # Get precip cube and append to list if successful
        rains = get_rains(scratch_d, member)
        if rains:
            rain_cubes.append(rains)

        # Get visibility cube and append to list if successful
        vis = get_vis(scratch_d, member)
        if vis:
           vis_cubes.append(vis)

        # Remove files from scratch directory
        os.system(f'rm {scratch_d}')
        os.system(f'rm {scratch_f}')

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
    """ 
    Wrapper function for allowing multiprocessing of a function and
    ensuring that the output is appended to a queue, to be picked up later.
    """
    queue.put(function(*args))


def get_mog_info(now_hour, hall):
    """
    Finds directory with latest mog-g files based on time now.
    """
    # Determine ow many hours to subtract to get latest model
    if now_hour.hour in [0, 6, 12, 18]:
        subtract_hours = 12
    elif now_hour.hour in [1, 7, 13, 19]:
        subtract_hours = 13
    elif now_hour.hour in [2, 8, 14, 20]:
        subtract_hours = 8
    elif now_hour.hour in [3, 9, 15, 21]:
        subtract_hours = 9
    elif now_hour.hour in [4, 10, 16, 22]:
        subtract_hours = 10
    else:
        subtract_hours = 11

    # Subtract hours
    g_hour = now_hour - timedelta(hours=subtract_hours)

    # Convert datetime to string in format of directory name
    dir_dt_str = g_hour.strftime('%Y%m%dT%H%MZ')

    # Full directory paths
    m_path = f'{MOG_G_DIR}/{dir_dt_str}/engl_um'
    full_path = f'{USER}@{hall}:{m_path}'

    # Determine members with long lead times
    long_members = []
    for member in range(45):
        mem_str = f'{member:03}'
        dirs = subprocess.Popen([f'ssh -Y {hall} ls '
                                 f'{m_path}/engl_um_{mem_str}/'],
                                stdout=subprocess.PIPE, shell=True)
        (out, err) = dirs.communicate()
        if 'englaa_pf150' in str(out):
            long_members.append(member)   

    # Determine lead time numbers used for filenames required
    f_nums = []
    for num in FNAME_NUMS:

        # 000 file only has T+0 in it
        if num == 0:

            # Check if withing start and end dates and append number if so
            if FIRST_DT <= g_hour <= LAST_DT:
                f_nums.append(num)
  
        # All other file have 3 lead times up to the lead time in the filename
        else:
            for lead in [int(num) - 2, int(num) - 1, int(num)]:

                # Get valid date
                vdt = g_hour + timedelta(hours=lead)

                # Check if file has any relevant valid dates in it and append to
                # list if so
                if FIRST_DT <= vdt <= LAST_DT:
                    f_nums.append(num)
                    break

    return g_hour, full_path, f_nums, long_members


def main(hall):
    """
    Copies files from HPC, extracts data and creates wind, temperature and
    relative humidity cubes. Probabilities are then calculated based on a few
    thresholds and cross-section plots are created and saved as png files.
    HTML page displaying plots is also updated.
    """
    # Make image directory if needed
    img_dir = f'{HTML_DIR}/images/{SITE_NAME.replace(" ", "_")}'
    if not os.path.exists(img_dir):
        os.system(f'mkdir {img_dir}')

    # Calculate period of forecast from first and last dts
    fcast_period = int((LAST_DT - FIRST_DT).total_seconds() / 3600) - 1

    # Time now (only to hour)
    now_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    # now_hour = datetime(2023, 6, 29, 7)

    # Directory path of latest MOG-G data and lead time fnames
    # g_hour, latest_mog_dir, f_nums, long_members = get_mog_info(now_hour, hall)
    g_hour = datetime(2023, 6, 30, 0)
    # latest_mog_dir = 'alanyon@xcfl01:/critical/opfc/suites-oper/global/share/cycle/20230628T1800Z/engl_um'
    # f_nums = ['015']
    # long_members = [0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

    # # To add cubes to
    # wind_spd_cube_list = iris.cube.CubeList([])
    # temp_cube_list = iris.cube.CubeList([])
    # rel_hum_cube_list = iris.cube.CubeList([])
    # rain_cube_list = iris.cube.CubeList([])
    # vis_cube_list = iris.cube.CubeList([])

    # # Parrelellise in 2 chunks - should be 18 ensemble members, so 9 at a time
    # out_lists = []
    # for mem_list in [long_members[:9], long_members[9:]]:

    #     # Use multiprocessing to process each hour in parellel
    #     queue = Queue()
    #     processes = []

    #     # out_list = []

    #     # Parrallelise for each ensemble member
    #     for member in mem_list:

    #         # out_list.append(data_from_files(latest_mog_dir, f_nums, member))

    #         # Add to processes list for multiprocessing
    #         args = (data_from_files, [latest_mog_dir, f_nums, member], queue)
    #         processes.append(Process(target=_mp_queue, args=args))

    #     # Start processes
    #     for process in processes:
    #         process.start()

    #     # Collect output from processes and close queue
    #     out_list = [queue.get() for _ in processes]
    #     queue.close()

    #     # Wait for all processes to complete before continuing
    #     for process in processes:
    #         process.join

    #     # Append output to list
    #     out_lists.append(out_list)

    # # Append output to cubelists
    # for out_list in out_lists:
    #     for item in out_list:
    #         wind_cubes, temp_cubes, rel_hum_cubes, rain_cubes, vis_cubes = item
    #         for wind_cube in wind_cubes:
    #             wind_spd_cube_list.append(wind_cube)
    #         for temp_cube in temp_cubes:
    #             temp_cube_list.append(temp_cube)
    #         for rel_hum_cube in rel_hum_cubes:
    #             rel_hum_cube_list.append(rel_hum_cube)
    #         for rain_cube in rain_cubes:
    #             rain_cube_list.append(rain_cube)
    #         for vis_cube in vis_cubes:
    #             vis_cube_list.append(vis_cube)

    # # Pickle/unpickle
    # uf.pickle_data([wind_spd_cube_list, temp_cube_list,
    #                 rel_hum_cube_list, rain_cube_list, vis_cube_list], 
    #                 f'{SCRATCH_DIR}/pickle')

    (wind_spd_cube_list, temp_cube_list, rel_hum_cube_list, rain_cube_list, 
        vis_cube_list) = uf.unpickle_data(f'{SCRATCH_DIR}/pickle')

    # Calculate probabilities and make cross-section plots
    # probs_and_plots(wind_spd_cube_list, 'wind', g_hour)
    # probs_and_plots(temp_cube_list, 'temp', g_hour)
    # probs_and_plots(rel_hum_cube_list, 'relative_humidity', g_hour)
    # rain_plots(rain_cube_list, g_hour)
    # vis_plots(vis_cube_list, g_hour)

    # Update HTML page
    date_str = g_hour.strftime('%Y%m%d%HZ')
    update_html(date_str)


if __name__ == "__main__":

    # Print time
    time_1 = uf.print_time('started')

    # If code fails, try changing HPC hall
    # try:
    main('xcfl01')
    print('xcfl hall used')
    # except Exception:
        # print('Changing hall')
        # main('xcel01')
        # print('xcel hall used')

    # Print time
    time_2 = uf.print_time('Finished')

    # Print time taken
    uf.time_taken(time_1, time_2, unit='minutes')
