from datetime import datetime, timedelta
import iris
import csv
import pickle
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
from math import sin, cos, asin, sqrt, radians


# Earth's radius in km
EARTH_RADIUS = 6373.0


def map_with_grid(f_name, degs_1=False, degs_2=False, kms_1=False, kms_2=False,
                  extent=False, proj=ccrs.PlateCarree(), labels=True):
    """
    Creates map (with gridlines if selected).
    """
    # number of lats/lons to plot grid
    if degs_1 or kms_1:
        if kms_1:
            degs_1 = kms / 111
        lons_1 = 360 / degs_1 + 1
        lats_1 = 180 / degs_1 + 1
    if degs_2 or kms_2:
        if kms_2:
            degs_2 = kms_2 / 111
        lons_2 = 360 / degs_2 + 1
        lats_2 = 180 / degs_2 + 1

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    # Create global map with coastlines, land, etc
    ax = plt.axes(projection=proj)
    ax.coastlines('10m')
    ax.stock_img()

    # Draw gridlines if resolution defined
    if degs_1:
        gls = ax.gridlines(proj, linestyle='-', color='r',
                           xlocs = np.linspace(-180, 180, lons_1),
                           ylocs= np.linspace(-90, 90, lats_1),
                           linewidth=1, draw_labels=labels)
        gls.xformatter = LONGITUDE_FORMATTER
        gls.yformatter = LATITUDE_FORMATTER
    if degs_2:
        gls = ax.gridlines(proj, linestyle='--', color='y',
                           xlocs = np.linspace(-180, 180, lons_2),
                           ylocs= np.linspace(-90, 90, lats_2),
                           linewidth=1, draw_labels=labels)
        gls.xformatter = LONGITUDE_FORMATTER
        gls.yformatter = LATITUDE_FORMATTER

    # Set extent if required
    if extent:
        ax.set_extent(extent, crs=proj)

    # Remove white space around plots
    plt.tight_layout()

    # Save figure
    fig.savefig(f_name)
    plt.close()


def print_time(time_str):
    """
    Prints a string with the current time and returns current datetime.
    """
    time_now = datetime.now()
    print(time_str, time_now)

    return time_now


def time_taken(time_1, time_2, unit='seconds'):
    """
    Prints time between two datetimes in given time unit.
    """
    # Difference in seconds
    time_diff_secs = (time_2 - time_1).total_seconds()

    # Convert to required units
    if unit == 'microseconds':
        time_units = time_diff_secs * 1000000
    elif unit == 'milliseconds':
        time_units = time_diff_secs * 1000
    elif unit == 'seconds':
        time_units = time_diff_secs
    elif unit == 'minutes':
        time_units = time_diff_secs / 60
    elif unit == 'hours':
        time_units = time_diff_secs / 3600
    else:
        print('Choose one of these time units next time: microseconds, '
              'milliseconds, seconds, minutes or hours. Defaulting to seconds')
        time_units = time_diff_secs
        unit = 'seconds'

    # Print nice string
    print('')
    print('Time taken to run: {} {}'.format(time_units, unit))
    print('')


def speed_from_u_v(u_winds, v_winds):
    """
    Calculates wind speed from u and v components.
    """

    wind_speed = (u_winds ** 2 + v_winds ** 2) ** 0.5

    return wind_speed

def epoch_to_dt(epoch_time, units='hours'):
    """
    Converts Epoch time (hours since 1970) to datetime.datetime object.
    """
    # Ensure epoch time is in seconds
    if units == 'hours':
        epoch_time_seconds = epoch_time * 3600
    elif units == 'seconds':
        epoch_time_seconds = epoch_time
    else:
        print('Epoch time must be in seconds or hours')
        return None

    # Convert to datetimes
    d_time = (datetime.utcfromtimestamp(epoch_time_seconds))

    return d_time


def dt_to_epoch(dt_time, units='hours'):
    """
    Converts Epoch time (hours since 1970) to datetime.datetime object.
    """
    # Convert to epoch seconds
    epoch_time_seconds = (datetime.timestamp(dt_time))

    # Ensure epoch time is in seconds
    if units == 'hours':
        epoch_time = epoch_time / 3600
    elif units == 'seconds':
        epoch_time = epoch_time
    else:
        print('Epoch time must be in seconds or hours')
        return None

    return epoch_time


def regrid_cube(cube, lons, lats):
    """
    Regrids cube based on list of longitudes and latitudes using linear
    interpolation.
    """
    reg_cube = cube.interpolate([('longitude', lons), ('latitude', lats)],
                                iris.analysis.Linear())

    return reg_cube


def dist_btw_pnts(lat1, lon1, lat2, lon2):
    """
    Calculates distance in km between two lat/lon points.
    """
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    # Radius of earth in kilometers is 6371
    km = EARTH_RADIUS* c

    return km

def best_bd_site(fname, lat, lon, height, radius):
    """
    Finds most appropriate BestData site, based on geographical location and
    site elevation.
    """
    sites = {}
    best_site = [None, 1000000, 1000000]

    with open(fname, errors='ignore') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            # Ensure correct information in row
            if (row[1][:4] == 'LAT=' and row[2][:4] == 'LON=' and
                    row[4][:4] == 'ALT='):
                if len(row[1]) > 4 and len(row[2]) > 4 and len(row[4]) > 4:

                    row_1 = row[1]
                    row_2 = row[2][4:]

                    site_name = row[3][10:].strip('\"')
                    site_code = int(row[0][22:])
                    bd_lat = float(row[1][4:])
                    bd_lon = float(row[2][4:])
                    bd_height = float(row[4][4:])
                    height_diff = abs(height - bd_height)
                    dist_from_site = dist_btw_pnts(lat, lon, bd_lat, bd_lon)


                    # Ammend to best_site if appropriate
                    if (dist_from_site < best_site[1] and height_diff < 50):
                        best_site = [site_name, dist_from_site, bd_height]

                    # Append to alternative sites list if criteria met
                    if dist_from_site <= radius and height_diff < 50:
                        sites[site_name] = [site_code, dist_from_site,
                                            bd_height]

    print('Best_site is {} at a distance of {:.2f}km away. Height of site is '
          '{:.2f}m, a difference of {:.2f}m from that at the point of '
          'interest.'.format(best_site[0], best_site[1], best_site[2],
                             abs(best_site[2] - height)))
    print('')
    print('Alternative sites:')

    for site in sites:
        if site == best_site[0]:
            sites[site].append('best')
        else:
            sites[site].append('not_best')
            print('{}: {:.2f}km away, height {:.2f}m, difference in '
                  'height {:.2f}m, site code {}'.format(
                        site, sites[site][1], sites[site][2],
                        abs(sites[site][2] - height), sites[site][0]
                        ))

    return sites


def dts_from_pandas(df_dts):
    """
    Converts pandas datetimes to normal datetimes.
    """

    dts = []
    for df_dt in df_dts:
        ts = ((df_dt - np.datetime64('1970-01-01T00:00:00Z'))
              / np.timedelta64(1, 's'))
        dt = datetime.utcfromtimestamp(ts)
        dts.append(dt)
    dts = np.array(dts)

    return dts


def pickle_data(data, fname):

    file_object = open(fname, 'wb')
    pickle.dump(data, file_object)
    file_object.close()


def unpickle_data(fname):

    with open(fname, 'rb') as file_object:
        unpickle = pickle.Unpickler(file_object)
        data = unpickle.load()

    return data
