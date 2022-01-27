"""
To do:

    Make sure Skyport units correct.
    Convert Skyports directions (negative??).
"""

import numpy as np
import csv
import sys
import glob
import matplotlib.pyplot as plt
from matplotlib import ticker
from datetime import datetime
from math import sqrt, atan2, pi
import iris
import iris.plot as iplt
from iris.analysis.cartography import rotate_pole
import useful_functions as uf

DATA_DIR = '/data/users/alanyon/uav/skyports'
SKY_FNAME = '{}/skyports.csv'.format(DATA_DIR)
P_FNAME = '{}/sky_pickle'.format(DATA_DIR)

UKV_M_FNAMES = ['{}/prodm_op_ukv_20210415_12_00{}.pp'.format(DATA_DIR, ind)
                for ind in [0, 2]]
UKV_S_FNAMES = ['{}/prods_op_ukv_20210415_12_00{}.pp'.format(DATA_DIR, ind)
                for ind in [0, 2]]
U_CON = iris.AttributeConstraint(STASH='m01s00i002')
V_CON = iris.AttributeConstraint(STASH='m01s00i003')
OROG_CON = iris.AttributeConstraint(STASH='m01s00i033')


def get_ukv_cubes(lat_lon_bounds):
    """
    Loads iris cubes and concatenates together.
    """
    # Load cubes
    u_cubes = [iris.load_cube(fname, U_CON) for fname in UKV_M_FNAMES]
    v_cubes = [iris.load_cube(fname, V_CON) for fname in UKV_M_FNAMES]

    # Concatenate cubes
    u_cube = iris.cube.CubeList(u_cubes).concatenate()[0]
    v_cube = iris.cube.CubeList(v_cubes).concatenate()[0]

    # Load orography cube
    orog_cube = iris.load(UKV_S_FNAMES[0], OROG_CON)[0]

    # Take intersections over area of interest
    min_lat, max_lat, min_lon, max_lon = lat_lon_bounds
    u_cube, v_cube, orog_cube = [cube.intersection(
        grid_longitude=(min_lon - 0.3, max_lon + 0.3),
        grid_latitude=(min_lat - 0.2, max_lat + 0.2)
        )
        for cube in [u_cube, v_cube, orog_cube]]

    # Define orography cubes as auxiliary coordinates
    orog_coord = iris.coords.AuxCoord(orog_cube.data, 'surface_altitude',
                                      units='m')

    # Add orography as auxiliary coordinate to u and v cubes
    u_cube.add_aux_coord(orog_coord, [2, 3])
    v_cube.add_aux_coord(orog_coord, [2, 3])

    # Define hybrid height coordinates
    factory_u = iris.aux_factory.HybridHeightFactory(
        u_cube.coord('level_height'),
        u_cube.coord('sigma'),
        u_cube.coord('surface_altitude')
        )
    factory_v = iris.aux_factory.HybridHeightFactory(
        v_cube.coord('level_height'),
        v_cube.coord('sigma'),
        v_cube.coord('surface_altitude')
        )

    # Add hybrid height coordinate to u and v cubes
    u_cube.add_aux_factory(factory_u)
    v_cube.add_aux_factory(factory_v)

    # Convert units from m/s to knot
    u_cube.convert_units('knots')
    v_cube.convert_units('knots')

    # Convert altitude units from metres to feet
    u_cube.coord('altitude').convert_units('feet')
    v_cube.coord('altitude').convert_units('feet')
    u_cube.coord('level_height').convert_units('feet')
    v_cube.coord('level_height').convert_units('feet')
    u_cube.coord('surface_altitude').convert_units('feet')
    v_cube.coord('surface_altitude').convert_units('feet')

    # Get wind speed cube
    wind_spd = (u_cube.data ** 2 + v_cube.data ** 2) ** 0.5
    wind_spd_cube = u_cube.copy(data=wind_spd)

    return u_cube, v_cube, wind_spd_cube, orog_cube


def get_mog_uk_cubes(lat_lon_bounds, orog_cube):
    """
    Loads MOGREPS-UK cubes and concatenates together.
    """
    # Min and max lats/lons of Skyports drone
    min_lat, max_lat, min_lon, max_lon = lat_lon_bounds

    # Define filenames
    fnames = glob.glob('{}/*mogreps*'.format(DATA_DIR))

    # Load cubes and get wind speeds
    (u_cubes_14, u_cubes_15, v_cubes_14,
     v_cubes_15, spd_cubes_14, spd_cubes_15) = [], [], [], [], [], []
    for fname in fnames:

        # Load cubes, taking intersection over area of interest
        u_cube = iris.load_cube(fname, U_CON)
        v_cube = iris.load_cube(fname, V_CON)

        u_cube = u_cube.intersection(grid_longitude=(min_lon - 0.3,
                                                     max_lon + 0.3),
                                     grid_latitude=(min_lat - 0.2,
                                                    max_lat + 0.2))
        v_cube = v_cube.intersection(grid_longitude=(min_lon - 0.3,
                                                     max_lon + 0.3),
                                     grid_latitude=(min_lat - 0.2,
                                                    max_lat + 0.2))

        # Regrid orography cube
        orog_reg = orog_cube.regrid(u_cube, iris.analysis.Linear())

        # Define orography cubes as auxiliary coordinates
        orog_coord = iris.coords.AuxCoord(orog_reg.data, 'surface_altitude',
                                          units='m')

        # Add orography as auxiliary coordinate to u and v cubes
        u_cube.add_aux_coord(orog_coord, [2, 3])
        v_cube.add_aux_coord(orog_coord, [2, 3])

        # Define hybrid height coordinates
        factory_u = iris.aux_factory.HybridHeightFactory(
            u_cube.coord('level_height'),
            u_cube.coord('sigma'),
            u_cube.coord('surface_altitude')
            )
        factory_v = iris.aux_factory.HybridHeightFactory(
            v_cube.coord('level_height'),
            v_cube.coord('sigma'),
            v_cube.coord('surface_altitude')
            )

        # Add hybrid height coordinate to u and v cubes
        u_cube.add_aux_factory(factory_u)
        v_cube.add_aux_factory(factory_v)

        # Convert units from m/s to knot
        u_cube.convert_units('knots')
        v_cube.convert_units('knots')

        # Convert altitude units from metres to feet
        u_cube.coord('altitude').convert_units('feet')
        v_cube.coord('altitude').convert_units('feet')
        u_cube.coord('level_height').convert_units('feet')
        v_cube.coord('level_height').convert_units('feet')
        u_cube.coord('surface_altitude').convert_units('feet')
        v_cube.coord('surface_altitude').convert_units('feet')

        # Get wind speed cube
        wind_spd = (u_cube.data ** 2 + v_cube.data ** 2) ** 0.5
        wind_spd_cube = u_cube.copy(data=wind_spd)

        # Get cubes valid at 1400Z and 1500Z and append to lists if found
        u_cubes_14, u_cubes_15 = get_14_15_cubes(u_cube, u_cubes_14,
                                                 u_cubes_15)
        v_cubes_14, v_cubes_15 = get_14_15_cubes(v_cube, v_cubes_14,
                                                 v_cubes_15)
        spd_cubes_14, spd_cubes_15 = get_14_15_cubes(wind_spd_cube,
                                                     spd_cubes_14,
                                                     spd_cubes_15)

    # Convert to probabilities based on wind speeds >= 27 knots
    prb_cube_14 = calc_probs(spd_cubes_14, 15)
    prb_cube_15 = calc_probs(spd_cubes_15, 15)

    # Collect into big list
    mog_data = [u_cubes_14, u_cubes_15, v_cubes_14, v_cubes_15, spd_cubes_14,
                spd_cubes_15, prb_cube_14, prb_cube_15]

    return mog_data


def calc_probs(cubes, threshold):
    """
    Returns cube with probabilities of wind speeds >= 27 knots.
    """
    # Calculate probabilities of winds >= 27 knots
    exceeds = [(cube.data >= threshold).astype(int) for cube in cubes]
    probs = sum(exceeds) / len(exceeds)

    # Create cube
    probs_cube = cubes[0].copy(data=probs)

    return probs_cube


def get_14_15_cubes(cube, cubes_14, cubes_15):
    """
    Collects cubes valid at 1400Z and 1500Z.
    """
    # Loop through all forecast times in cube
    for ind, epoch_time in enumerate(cube.coord('time').points):

        # Convert to datetime
        vdt = uf.epoch_to_dt(epoch_time)

        # Append to lists if appropriate
        if vdt == datetime(2021, 4, 15, 14):
            cubes_14.append(cube[ind])
        elif vdt == datetime(2021, 4, 15, 15):
            cubes_15.append(cube[ind])

    return cubes_14, cubes_15


def get_sky_data():
    """
    Reads required data from csv file, converting standard latitudes and
    longitudes to rotated pole coordinates in UKV cube.
    """
    # Load a cube to get pole coordinates from
    cube = iris.load_cube(UKV_M_FNAMES[0], U_CON)

    # Get rotated poles from one of the cubes
    pole_lon = cube.coord_system().grid_north_pole_longitude
    pole_lat = cube.coord_system().grid_north_pole_latitude

    # To append data to
    sky_data, sky_pos = [], []

    # Open csv file and take required info from each row
    count = 0
    with open(SKY_FNAME) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            # Miss first row and get data every 30 seconds
            if count > 1 and count % 60 == 0:
                # Collect required info
                time_lat_lon_alt_winddir_windspd = [float(row[ind]) for ind
                                                    in [2, 6, 8, 4, 15, 16]]
                sky_data.append(time_lat_lon_alt_winddir_windspd)
            # Get altitude and position every second
            elif count > 1 and count % 2 == 0:
                sky_pos.append([float(row[2]), float(row[6]), float(row[8]),
                                float(row[4])])
            count += 1

    # Convert list to numpy array and get lats/lons/times out
    sky_data = np.array(sky_data)
    times, lats, lons = [sky_data[:, ind] for ind in range(3)]
    sky_pos = np.array(sky_pos)
    times_all, lats_all, lons_all = [sky_pos[:, ind] for ind in range(3)]

    # Convert epoch time to hours instead of seconds
    sky_data[:, 0] = times / 3600
    sky_pos[:, 0] = times_all / 3600

    # Convert Skyport lats/lons to rotated pole coords
    sky_data[:, 2], sky_data[:, 1] = rotate_pole(lons, lats, pole_lon,
                                                 pole_lat)
    sky_pos[:, 2], sky_pos[:, 1] = rotate_pole(lons_all, lats_all, pole_lon,
                                               pole_lat)

    # Change directions to 0-360 degrees
    sky_data[:, 4][sky_data[:, 4] < 0] += 360

    # Lats/lons/altitude of drone
    sky_lats, sky_lons = sky_pos[:, 1], sky_pos[:, 2]

    # Get min and max lats/lons of Skyports drone
    min_lat, max_lat = np.min(sky_lats), np.max(sky_lats)
    min_lon, max_lon = np.min(sky_lons), np.max(sky_lons)
    lat_lon_bounds = [min_lat, max_lat, min_lon, max_lon]

    return sky_data, sky_pos, lat_lon_bounds


def ukv_wind(sky_time, lat, lon, altitude, u_cube, v_cube):
    """
    Gets UKV wind data based on time, latitude, longitude and altitude.
    """
    # Linearly interpolate to get U and V components from UKV at all levels
    sample_points = [('grid_latitude', lat), ('grid_longitude', lon),
                     ('time', sky_time)]
    u_wind_levs = u_cube.interpolate(sample_points, iris.analysis.Linear())
    v_wind_levs = v_cube.interpolate(sample_points, iris.analysis.Linear())

    # Interpolate between UKV model level altitudes using drone altitude
    alt_point = [('altitude', altitude)]
    u_wind = u_wind_levs.interpolate(alt_point, iris.analysis.Linear())
    v_wind = v_wind_levs.interpolate(alt_point, iris.analysis.Linear())

    # Get surface altitude
    surf_alt = u_wind.coord('surface_altitude').points[0]

    # Get values from u_wind and v_wind u_cubes
    u_wind, v_wind = u_wind.data, v_wind.data

    # Calculate wind speed and direction from U and V components
    wind_spd = sqrt(u_wind ** 2 + v_wind ** 2)
    wind_dir = atan2(u_wind / wind_spd, v_wind / wind_spd) * 180 / pi + 180

    return wind_dir, wind_spd, surf_alt


def mog_x_cubes(sky_time, lat, lon, altitude, mog_data):

    # Merge prob cubes
    prob_cube = iris.cube.CubeList(mog_data[6:]).merge()[0]

    # Linearly interpolate at sample points
    sample_points = [('grid_latitude', lat), ('grid_longitude', lon),
                     ('time', sky_time)]
    int_cube = prob_cube.interpolate(sample_points, iris.analysis.Linear())

    # Create coordinate for drone time
    sky_time_coord = iris.coords.AuxCoord(sky_time, long_name='sky_time')

    # Add drone time coordinate to cube
    int_cube.add_aux_coord(sky_time_coord)

    return int_cube


def make_plots(data):
    """
    Makes plots comparing Skyports winds to UKV winds.
    """
    # Unpack data array into separate components
    (times_epoch, _, _, altitude, wind_dir_sky, wind_spd_sky,
     wind_dir_ukv, wind_spd_ukv, surf_alt_ukv) = [data[:, ind]
                                                  for ind in range(9)]

    # Convert times to datetime objects
    vdts = [uf.epoch_to_dt(time_epoch) for time_epoch in times_epoch]

    # Make nicely formated date and time strings
    dt_strs = [vdt.strftime('%d/%m/%Y\n%H%MZ') for vdt in vdts]

    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))

    # Plot some lines
    axs[0, 0].plot(dt_strs, wind_spd_sky, color='r',
                   label='Skyports wind speed')
    axs[0, 0].plot(dt_strs, wind_spd_ukv, color='b', label='UKV wind speed')
    axs[0, 0].set_ylabel('Wind speed (knots)')
    axs[0, 1].plot(dt_strs, wind_dir_sky, color='r',
                   label='Skyports wind direction')
    axs[0, 1].plot(dt_strs, wind_dir_ukv, color='b',
                   label='UKV wind direction')
    axs[0, 1].set_ylabel('Wind direction (degrees)')
    axs[1, 0].plot(dt_strs, altitude, color='g',
                   label='Skyports altitude (above sea level)')
    axs[1, 0].set_ylabel('Altitude (feet)')
    axs[1, 0].plot(dt_strs, surf_alt_ukv, color='brown',
                   label='UKV surface altitude')
    axs[1, 0].set_ylabel('Altitude (feet)')
    axs[1, 1].plot(dt_strs, altitude - surf_alt_ukv, color='magenta',
                   label='Skyports altitude - UKV surface altitude')
    axs[1, 1].set_ylabel('Altitude (feet)')

    # Set major ticks for x axis
    major_xticks = np.arange(0, len(dt_strs), 12)
    minor_xticks = np.arange(0, len(dt_strs), 1)

    # Stuff common to both axes
    for ax in axs.ravel():
        ax.set_xlabel('Date and time')
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % 12 != 0:
                label.set_visible(False)
        # ax.set_xticks(major_xticks)
        # ax.set_xticks(minor_xticks, minor = True)
        ax.grid(color='grey', axis='y')

        # ax.set_xticks(xtick_locs)
        # ax.set_xticklabels(xlabels, fontsize=8)
        ax.legend(loc='best')

    # Figure title
    fig.suptitle('Comparison of Skyports and UKV winds')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure and close plot
    fname = '{}/plots/wind_spd_dir.png'.format(DATA_DIR)
    fig.savefig(fname)
    plt.close()


def make_overlays(sky_pos, u_cube, v_cube, spd_cube):
    """
    Makes gridded model data maps with path of drone overlain.
    """
    # Lats/lons/altitude of drone
    sky_lats, sky_lons, sky_alts = sky_pos[:, 1], sky_pos[:, 2], sky_pos[:, 3]

    # Get cubes valid at 14Z and 15Z at 3 model levels
    spd_cubes = [spd_cube[2][4], spd_cube[2][5], spd_cube[2][6],
                 spd_cube[3][4], spd_cube[3][5], spd_cube[3][6]]
    u_cubes = [u_cube[2][4], u_cube[2][5], u_cube[2][6],
               u_cube[3][4], u_cube[3][5], u_cube[3][6]]
    v_cubes = [v_cube[2][4], v_cube[2][5], v_cube[2][6],
               v_cube[3][4], v_cube[3][5], v_cube[3][6]]

    # Ensure each plot uses same wind colour scale
    max_wind = max([np.max(cube.data.flatten()) for cube in spd_cubes])
    min_wind = min([np.min(cube.data.flatten()) for cube in spd_cubes])
    levels = np.linspace(min_wind, max_wind, 100)

    # Create figure
    fig, axs = plt.subplots(2, 3, figsize=(12, 10))

    # Make plots
    for ax, spd_cube, u_cube, v_cube in zip(axs.ravel(), spd_cubes, u_cubes,
                                            v_cubes):
        # Get details from cube
        fcast_time_epoch = spd_cube.coord('time').points[0]
        fcast_time = uf.epoch_to_dt(fcast_time_epoch)
        level_height = int(spd_cube.coord('level_height').points[0])
        lead = int(spd_cube.coord('forecast_period').points[0])

        # Define coarse res lat/lon grids for quiver arrows
        lon = u_cube.coord('grid_longitude')
        lat = u_cube.coord('grid_latitude')
        lon_min, lat_min = lon.points.min(), lat.points.min()
        lon_max, lat_max = lon.points.max(), lat.points.max()
        lon_low = np.arange(lon_min, lon_max, 0.05)
        lat_low = np.arange(lat_min, lat_max, 0.05)

        # Regrid winds to coarse resolution for quiver arrows
        u_cube_low = u_cube.interpolate([('grid_longitude', lon_low),
                                         ('grid_latitude', lat_low)],
                                        iris.analysis.Linear())
        v_cube_low = v_cube.interpolate([('grid_longitude', lon_low),
                                         ('grid_latitude', lat_low)],
                                        iris.analysis.Linear())

        # Ensure correct axes used
        plt.sca(ax)
        # Draw plot
        contours = iplt.contourf(spd_cube, levels=levels, cmap='YlGnBu')
        # Get rid of contour lines
        for contour in contours.collections:
            contour.set_edgecolor("face")
        # Add coastlines
        plt.gca().coastlines('10m')
        # Ensure square grid cells
        ax.set_aspect('equal')
        # Plot drone track
        alts = plt.scatter(sky_lons, sky_lats, c=sky_alts, cmap='Reds', s=10)
        # Add arrows to show the wind vectors
        # Get the coordinate reference system used by the quiver data
        u_lon = u_cube_low.coord('grid_longitude')
        trans = u_lon.coord_system.as_cartopy_projection()

        plt.quiver(lon_low, lat_low, u_cube_low.data, v_cube_low.data,
                   pivot='middle', transform=trans)
        # Add subplot title
        plt.title('Model level height: ~{}ft,\nForecast time: {}\n'
                  'Lead time: {} hours'.format(level_height, fcast_time, lead))

    # Add colour bars
    plt.subplots_adjust(bottom=0.15)
    cbaxes_wind = fig.add_axes([0.12, 0.06, 0.78, 0.02])
    cbaxes_alt = fig.add_axes([0.12, 0.13, 0.78, 0.02])
    cbar_wind = plt.colorbar(contours, cax=cbaxes_wind,
                             orientation='horizontal')
    cbar_alt = plt.colorbar(alts, cax=cbaxes_alt, orientation='horizontal')
    tick_locator = ticker.MaxNLocator(nbins=8)
    cbar_wind.locator = tick_locator
    cbar_wind.update_ticks()
    tick_locator = ticker.MaxNLocator(nbins=8)
    cbar_alt.locator = tick_locator
    cbar_alt.update_ticks()
    cbar_wind.ax.tick_params(labelsize=10)
    cbar_wind.set_label('Wind speed (knots)', fontsize=10)
    cbar_alt.ax.tick_params(labelsize=10)
    cbar_alt.set_label('Drone altitude above sea level (feet)', fontsize=10)

    # Figure title
    fig.suptitle('UKV wind speeds with drone path overlain', fontsize=25)

    # Save figure and close plot
    fname = '{}/plots/overlays.png'.format(DATA_DIR)
    fig.savefig(fname)
    plt.close()


def mog_plots(mog_data, sky_pos):
    """
    Makes MOGREPS-UK plots.
    """
    # Unpack list
    u_14, u_15, v_14, v_15, spd_14, spd_15, prb_14, prb_15 = mog_data

    # Get cubes 3 model levels
    prb_14_15_levs = [prb_14[4], prb_14[5], prb_14[6], prb_15[4], prb_15[5],
                      prb_15[6]]
    # Change fractions to percentages
    prb_14_15_levs = [cube * 100 for cube in prb_14_15_levs]

    # Lats/lons/altitude of drone
    sky_lats, sky_lons, sky_alts = sky_pos[:, 1], sky_pos[:, 2], sky_pos[:, 3]

    # Colours and probability levels used in plot
    levels = [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 100]
    colors = ['#ffffff', '#e6ffe6', '#ccffcc', '#b3ffb3', '#99ff99', '#80ff80',
              '#80bfff', '#4da6ff', '#1a8cff', '#0073e6', '#0059b3', '#004080',
              '#004080']

    # Create figure
    fig, axs = plt.subplots(2, 3, figsize=(12, 10))

    # Make plots
    for ax, cube in zip(axs.ravel(), prb_14_15_levs):

        # Get details from cube
        fcast_time_epoch = cube.coord('time').points[0]
        fcast_time = uf.epoch_to_dt(fcast_time_epoch)
        level_height = int(cube.coord('level_height').points[0])

        # Ensure correct axes used
        plt.sca(ax)
        # Draw plot
        contours = iplt.contourf(cube, levels=levels, colors=colors)
        # Get rid of contour lines
        for contour in contours.collections:
            contour.set_edgecolor("face")
        # Add coastlines
        plt.gca().coastlines('10m')
        # Ensure square grid cells
        ax.set_aspect('equal')
        # Plot drone track
        alts = plt.scatter(sky_lons, sky_lats, c=sky_alts, cmap='Reds', s=10)

        # Add subplot title
        plt.title('Model level height: ~{}ft,\nForecast time: '
                  '{}'.format(level_height, fcast_time))

    # Add colour bars
    plt.subplots_adjust(bottom=0.15)
    cbaxes_probs = fig.add_axes([0.12, 0.06, 0.78, 0.02])
    cbaxes_alt = fig.add_axes([0.12, 0.13, 0.78, 0.02])
    cbar_probs = plt.colorbar(contours, cax=cbaxes_probs,
                              orientation='horizontal')
    cbar_alt = plt.colorbar(alts, cax=cbaxes_alt, orientation='horizontal')
    cbar_probs.set_ticks(levels)
    cbar_probs.set_ticklabels(['{}%'.format(perc) for perc in levels])
    tick_locator = ticker.MaxNLocator(nbins=8)
    cbar_alt.locator = tick_locator
    cbar_alt.update_ticks()
    cbar_probs.ax.tick_params(labelsize=10)
    cbar_probs.set_label('Probability of winds exceeding 15 knots',
                         fontsize=10)
    cbar_alt.ax.tick_params(labelsize=10)
    cbar_alt.set_label('Drone altitude above sea level (feet)', fontsize=10)

    # Figure title
    fig.suptitle('MOGREPS-UK probabilities with drone path overlain',
                 fontsize=25)

    # Save figure and close plot
    fname = '{}/plots/mogreps.png'.format(DATA_DIR)
    fig.savefig(fname)
    plt.close()


def mog_x_plots(mog_x_cube, sky_pos):
    """
    Makes cross-section probability plots along drone path
    """
    # Lats/lons/altitude of drone
    (sky_times, sky_lats,
     sky_lons, sky_alts) = (sky_pos[:, 0], sky_pos[:, 1],
                            sky_pos[:, 2], sky_pos[:, 3])

    # Convert to datetime strings
    sky_times_dt = [uf.epoch_to_dt(sky_time).strftime('%d/%m/%Y\n%H%MZ')
                    for sky_time in sky_times]

    # Change fractions to percentages
    mog_x_cube = mog_x_cube * 100
    # mog_x_15_cube = mog_x_15_cube * 100

    # Colours and probability levels used in plot
    levels = [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 100]
    colors = ['#ffffff', '#e6ffe6', '#ccffcc', '#b3ffb3', '#99ff99', '#80ff80',
              '#80bfff', '#4da6ff', '#1a8cff', '#0073e6', '#0059b3', '#004080',
              '#004080']

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Make plots
    # Get cross-section from cube
    cross_section = next(mog_x_cube.slices(['sky_time',
                                            'model_level_number']))

    # Draw plot
    contours = iplt.contourf(cross_section, levels=levels, colors=colors,
                             coords=['sky_time', 'altitude'])
    # Get rid of contour lines
    for contour in contours.collections:
        contour.set_edgecolor("face")

    # Plot drone track
    alts = plt.scatter(sky_times, sky_alts, c='r', s=10)

    # Set limits and labels
    ax.set_ylim(0, np.max(sky_alts) * 1.1)
    ax.set_ylabel('Altitude (ft)')

    # For x axis labels
    plt.xticks(sky_times[0::500], sky_times_dt[0::500])

    # Add colour bar
    plt.subplots_adjust(bottom=0.23)
    cbaxes_probs = fig.add_axes([0.12, 0.1, 0.78, 0.02])
    cbar_probs = plt.colorbar(contours, cax=cbaxes_probs,
                              orientation='horizontal')
    cbar_probs.set_ticks(levels)
    cbar_probs.set_ticklabels(['{}%'.format(perc) for perc in levels])
    cbar_probs.ax.tick_params(labelsize=10)
    cbar_probs.set_label('Probability of wind exceeding 15 knots',
                         fontsize=12)

    # Figure title
    fig.suptitle('MOGREPS-UK probabilities - cross-section over drone path',
                 fontsize=25)

    # Save figure and close plot
    fname = '{}/plots/mogreps_x_section.png'.format(DATA_DIR)
    fig.savefig(fname)
    plt.close()


def main(new_data):

    # For Pickle filenames
    mog_comps = ['u_14', 'u_15', 'v_14', 'v_15', 'spd_14', 'spd_15', 'prb_14',
                 'prb_15']

    # Only get new data if necessary due to time taken to get it
    if new_data == 'yes':

        # Load skyports data, converting lats/lons to UKV rotated pole
        # coordinates
        sky_data, sky_pos, lat_lon_bounds = get_sky_data()

        # Load model level UKV and MOGREPS-UK wind cubes
        (u_cube, v_cube,
         wind_spd_cube, orog_cube) = get_ukv_cubes(lat_lon_bounds)

        # Get MOGREPS-UK data, separating into forecasts valid at 1400Z and
        # 1500Z
        mog_data = get_mog_uk_cubes(lat_lon_bounds, orog_cube)

        # Pickle data
        for mog, comp in zip(mog_data, mog_comps):
            uf.pickle_data(mog, '{}/mog_{}_pickle'.format(DATA_DIR, comp))

        # Compare Skyports data to UKV
        sky_ukv_data = []
        mog_xs = []
        for time_epoch, lat, lon, altitude, wind_dir, wind_spd in sky_data:

            # Find equivalent wind forecast in UKV
            (wind_dir_ukv,
             wind_spd_ukv, surf_alt_ukv) = ukv_wind(time_epoch, lat, lon,
                                                    altitude, u_cube, v_cube)
            sky_ukv_data.append([time_epoch, lat, lon, altitude, wind_dir,
                                 wind_spd, wind_dir_ukv, wind_spd_ukv,
                                 surf_alt_ukv])

            # Get mogreps-uk cross section cubes
            mog_x = mog_x_cubes(time_epoch, lat, lon, altitude, mog_data)
            mog_xs.append(mog_x)

        sky_ukv_data = np.array(sky_ukv_data)

        mog_x_cube = iris.cube.CubeList(mog_xs).merge()[0]
        # mog_x_15_cube = iris.cube.CubeList(mog_x_15s).merge()[0]

        # Pickle data for later use
        data_pos_cubes = [sky_ukv_data, sky_pos, u_cube, v_cube, wind_spd_cube,
                          mog_x_cube]
        uf.pickle_data(data_pos_cubes, P_FNAME)

    # Otherwise unpickle data
    else:
        mog_data = [uf.unpickle_data('{}/mog_{}_pickle'.format(DATA_DIR, comp))
                    for comp in mog_comps]
        (sky_ukv_data, sky_pos, u_cube, v_cube, wind_spd_cube,
         mog_x_cube) = uf.unpickle_data(P_FNAME)

    # Make some plots
    mog_plots(mog_data, sky_pos)
    mog_x_plots(mog_x_cube, sky_pos)
    make_plots(sky_ukv_data)
    make_overlays(sky_pos, u_cube, v_cube, wind_spd_cube)


if __name__ == "__main__":

    # Print time
    time_1 = uf.print_time('Started')

    try:
        new_data = sys.argv[1]
    except:
        print('WARNING! Arguments not set correctly so will exit python '
              'script')

    main(new_data)

    # Print time
    time_2 = uf.print_time('Finished')

    # Print time taken
    uf.time_taken(time_1, time_2)
