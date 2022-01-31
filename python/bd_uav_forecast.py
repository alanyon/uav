import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
# Local import
import useful_functions as uf

# Import environment variables
try:
    USER = os.environ['USER']
    BEST_DATA_DIR = os.environ['BEST_DATA_DIR']
    SCRATCH_DIR = os.environ['SCRATCH_DIR']
    HTML_DIR = os.environ['HTML_DIR']
    DATA_FILE = os.environ['DATA_FILE']
    START_DATE_TIME = os.environ['START_DATE_TIME']
    START_DATE = os.environ['START_DATE']
    START_TIME = os.environ['START_TIME']
    URL_START = os.environ['URL_START']
    SIDEBAR = os.environ['SIDEBAR']
except KeyError as err:
    raise IOError('Environment variable {} not set.'.format(str(err)))

# For converting from m/s to mph and mph to knots
MS_TO_KTS = 1.94384
MPH_TO_KTS = 0.86897423357831

# ==============================================================================
# Change these bits for new trial site/date
TRIAL_SITES = ['National Grid']
SITE_LATS = [53.145556]
SITE_LONS = [-0.991389]
TRIAL_HEIGHTS = [77]  # metres
FIRST_DTS = [datetime(2022, 2, 14, 0)]  # Year, month, day, hour
LAST_DTS = [datetime(2022, 2, 19, 1)]  # Year, month, day, hour
# ==============================================================================

# Shouldn't have to change any of the following but can if necessary
RADIUS_LIMIT = 5  # kilometres
# Parameter thresholds
TEMP_THRESHOLDS = [0, 25, 30]
# Wind thresholds given in mph but need to be converted to knots
MEAN_THRESHOLDS = [10 * MPH_TO_KTS, 15 * MPH_TO_KTS, 20 * MPH_TO_KTS]
GUST_THRESHOLDS = [20, 25 * MPH_TO_KTS, 30 * MPH_TO_KTS]
REL_HUM_THRESHOLDS = [40, None, 95]
VIS_THRESHOLDS = [1000, 500, 200]
# Best data filename
BD_FILE = '{}/bd_file.csv'.format(SCRATCH_DIR)
# Columns needed from best date csv file
USE_COLS = ['site', 'forecast_time', 'dry_bulb_temp', 'wind_speed',
            'wind_gust', 'wind_direction', 'visibility', 'sig_wx', 'low_cld',
            'med_cld', 'high_cld', 'tot_cld', 'precip_rate',
            'relative_humidity']
# For assigning names to columns in csv file
ls = []
for ind in range(1, 36):
    ls.append(str(ind))
COL_HEADS = ls
COL_HEADS[0:1] = ['site', 'forecast_time']
COL_HEADS[3:8] = ['dry_bulb_temp', 'wind_speed', 'wind_direction', 'wind_gust',
                  'visibility', 'relative_humidity']
COL_HEADS[10] = 'sig_wx'
COL_HEADS[23:26] = ['low_cld', 'med_cld', 'high_cld', 'tot_cld']
COL_HEADS[31] = 'precip_rate'


def get_bd_df(bd_sites, trial_site, first_dt, last_dt):
    """
    Reads Best Data csv file, filters data into Pandas dataframes and creates
    some plots.
    """
    # Copy files on HPC to scratch directory, removing previously used file
    hpc_bd_file = (f'{USER}@xcel01:{BEST_DATA_DIR}/{START_DATE}T{START_TIME}'
                   f'00Z/bd_main/SSPA_BEST_FCS_HOURLY_{START_DATE_TIME}_*.csv')
    os.system(f'rm {BD_FILE}')
    os.system(f'scp {hpc_bd_file} {BD_FILE}')

    # If no file found, try changing HPC hall
    if not os.path.exists(BD_FILE):
        hpc_bd_file = hpc_bd_file.replace('xcel', 'xcfl')
        os.system(f'scp {hpc_bd_file} {BD_FILE}')

    # Make directories for web page if using new trial site
    trl_img_dir = f'{HTML_DIR}/images/{trial_site.replace(" ", "_")}'
    if not os.path.exists(trl_img_dir):
        os.system(f'mkdir {trl_img_dir}')

    # Limit number of columns
    pd.set_option('display.max_columns', 50)

    # For changing date formats
    dateparse = lambda x: pd.datetime.strptime(x, '%d-%m-%Y%H:%M')

    # Create dataframe from csv file data
    bd_df = pd.read_csv(BD_FILE,
                        index_col=False,
                        names=COL_HEADS,
                        usecols=USE_COLS,
                        parse_dates=['forecast_time'],
                        date_parser=dateparse,
                        engine='c')

    # Convert values to numbers where appropriate
    for column in bd_df:
        if column not in ['site', 'forecast_time']:
            bd_df[column] = pd.to_numeric(bd_df[column], errors='coerce')

    # Filter for each site
    for site_name, site_list in bd_sites.items():
        # Unpack list
        site_code, site_dist, site_height, _ = site_list

        # Filter dataframe to get site data
        site_df = bd_df.loc[bd_df['site'] == site_code]

        # Get datetimes
        dts = uf.dts_from_pandas(site_df['forecast_time'].values)

        # Select for required dates
        dates_select = np.where((dts >= first_dt) & (dts < last_dt))

        # Get values, converting winds to knots, only for required dates
        new_dts = dts[dates_select]

        temps = site_df['dry_bulb_temp'].values[dates_select]
        wind_means = ((site_df['wind_speed'].values * MS_TO_KTS)[dates_select])
        wind_gusts = ((site_df['wind_gust'].values * MS_TO_KTS)[dates_select])
        wind_dirs = site_df['wind_direction'].values[dates_select]
        visibilities = site_df['visibility'].values[dates_select]
        precip_rates = site_df['precip_rate'].values[dates_select]
        low_cld = site_df['low_cld'].values[dates_select]
        med_cld = site_df['med_cld'].values[dates_select]
        high_cld = site_df['high_cld'].values[dates_select]
        rel_hum = site_df['relative_humidity'].values[dates_select]

        # Labels for plotting
        cld_labels = ['Low cloud', 'Medium cloud', 'High cloud']
        name = site_name.replace(' ', '_')

        # Make some plots (threholds in mph)
        make_plot(new_dts, temps, 'deg C', 'Dry Bulb Temperature', name,
                  site_dist, site_height, trial_site,
                  thresholds=TEMP_THRESHOLDS)
        make_plot(new_dts, precip_rates, 'mm/hr', 'Precipitation Rate', name,
                  site_dist, site_height, trial_site)
        make_plot(new_dts, wind_means, 'knots', 'Wind means', name, site_dist,
                  site_height, trial_site, thresholds=MEAN_THRESHOLDS)
        make_plot(new_dts, wind_gusts, 'knots', 'Wind gusts', name, site_dist,
                  site_height, trial_site, thresholds=GUST_THRESHOLDS)
        make_plot(new_dts, wind_dirs, 'degrees', 'Wind directions', name,
                  site_dist, site_height, trial_site)
        make_plot(new_dts, rel_hum, '%', 'Relative humidity', name, site_dist,
                  site_height, trial_site, thresholds=REL_HUM_THRESHOLDS)
        make_plot(new_dts, visibilities, 'metres', 'Visibility', name,
                  site_dist, site_height, trial_site,
                  thresholds=VIS_THRESHOLDS)
        make_plot(new_dts, [low_cld, med_cld, high_cld], 'Oktas', 'Cloud',
                  name, site_dist, site_height, trial_site, multiple=True,
                  labels=cld_labels)


def make_plot(dts, values, y_label, param, name, dist, height, trial_site,
              multiple=False, two_scales=False, labels=[],
              thresholds=[None, None, None]):
    """
    Creates and saves a line plot.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 7))

    # Make nicely formated date and time strings
    date_strings = [(f'{dts[ind].day:02d}/{dts[ind].month:02d}\n'
                     f'{dts[ind].hour:02d}Z')
                    for ind in range(len(dts))]

    # Gap between x ticks, depending on number of days shown on plot
    if len(date_strings) <= 72:
        gap = 3
    else:
        gap = 6

    # Define x axis ticks and labels
    xtick_locs, xlabels = [], []
    for ind, date in enumerate(date_strings):
        if ind % gap == 0:
            xtick_locs.append(ind)
            xlabels.append (date)

    # Plot line(s)
    if multiple:
        colours = ['r', 'b', 'g']
        for ind, (value_array, colour, label) in enumerate(zip(values,
                                                               colours,
                                                               labels)):
            if two_scales and ind == 2:
                ax2 = ax.twinx()
                lns_2 = ax2.plot(date_strings, value_array, color=colour,
                                 label=label)
                ax2.set_ylabel(y_label[1])
                # Add to lines for legend
                lns += lns_2
            else:
                lns_1 = ax.plot(date_strings, value_array, color=colour,
                                label=label)
                if two_scales:
                    ax.set_ylabel(y_label[0])
                else:
                    ax.set_ylabel(y_label)
                # Add to lines for legend
                if ind == 0:
                    lns = lns_1
                else:
                    lns += lns_1
        # Create legend
        labs = [ln.get_label() for ln in lns]
        legend = ax.legend(lns, labs, loc='upper right',
                           bbox_to_anchor=(1.2, 1, 0, 0))

    else:
        ax.plot(date_strings, values)
        ax.set_ylabel(y_label)

        # Plot fixed line thresholds
        if param == 'Dry Bulb Temperature':
            colours = ['r', 'orange', 'r']
            labels = ['Must not encounter', 'Cautionary', '']
        else:
            colours = ['g', 'orange', 'r']
            labels = ['Safe', 'Cautionary', 'Must not encounter']
        for thresh, colour, label in zip(thresholds, colours, labels):
            if thresh is not None:
                xlims = ax.get_xlim()
                ax.hlines(thresh, ax.get_xlim()[0], ax.get_xlim()[1],
                          colors=colour, label=label)
                ax.set_xlim(xlims)

    # Format plot
    ax.grid(color='grey', axis='x')
    ax.legend(loc='best')
    ax.set_xticks(xtick_locs)
    ax.set_xticklabels(xlabels, fontsize=8)
    title = (f'{param}. Elevation of site: {int(height)} m. Distance from '
             f'{trial_site}: {dist:.2f}km')
    ax.set_title(title)
    plt.tight_layout()

    # Save figure and close plot
    fname = (f'{HTML_DIR}/images/{trial_site.replace(" ", "_")}/{name}_'
             f'{param.replace(" ", "")}_{START_DATE_TIME}Z.png')
    fig.savefig(fname)
    plt.close()


def update_html(bd_sites, trial_site, trial_height):
    """
    Updates html file.
    """
    # File name of html file
    trial_fname = trial_site.replace(' ', '_')
    html_fname = (f'{HTML_DIR}/html/{trial_fname}_bd_fcasts.shtml')

    # Make new directories/files if needed
    if not os.path.exists(html_fname):

        # Make html file starting with template
        template = f'{HTML_DIR}/html/bd_template.shtml'
        os.system(f'cp {template} {html_fname}')

        # Put in trial-specific stuff
        file = open(html_fname, 'r')
        lines = file.readlines()
        file.close()
        first_lines = lines[:35]
        second_lines = lines[35:54]
        last_lines = lines[54:]

        first_lines[-1] = first_lines[-1].replace('TRIAL', trial_fname)
        second_lines[14] = second_lines[14].replace('TRIAL', trial_site)
        second_lines[14] = second_lines[14].replace('HEIGHT',
                                                    str(trial_height))
        for ind, site in enumerate(bd_sites):
            if bd_sites[site][3] == 'best':
                first_site = site.replace(' ', '_')
                second_lines.append('                        <option '
                                    f'selected="selected" value="{first_site}"'
                                    f'>{site}</option>\n')
            else:
                next_site = site.replace(' ', '_')
                second_lines.append('                        <option '
                                    f'value="{next_site}">{site}</option>\n')
        last_lines[5] = last_lines[5].replace('DATE', START_DATE_TIME)
        last_lines[21] = last_lines[21].replace('TRIAL', trial_fname)
        last_lines[21] = last_lines[21].replace('NAME', trial_site)
        last_lines[-11] = last_lines[-11].replace('TRIAL', trial_fname)
        last_lines[-11] = last_lines[-11].replace('SITE', first_site)
        last_lines[-11] = last_lines[-11].replace('DATE', START_DATE_TIME)

        # Concatenate the lists together
        new_lines = first_lines + second_lines + last_lines

        # Add site to sidebar
        file = open(SIDEBAR, 'r')
        lines = file.readlines()
        file.close()
        first_lines = lines[:-5]
        last_lines = lines[-5:]
        url = f'{URL_START}/{trial_fname}_bd_fcasts.shtml'
        first_lines.append(f'          <li><a href="{url}">{trial_site} '
                           'Best Data Forecasts</a></li>\n')
        # Concatenate the lists together
        side_lines = first_lines + last_lines

        # Re-write the lines to a new file
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
        first_lines = lines[:-35]
        last_lines = lines[-35:]

        # Edit html file and append/edit the required lines
        first_lines[-1] = first_lines[-1].replace(' selected="selected"', '')
        first_lines.append('                        '
                           '<option selected="selected" '
                           f'value="{START_DATE_TIME}Z">{START_DATE_TIME}Z'
                           '</option>\n')
        last_lines[-11] = last_lines[-11].replace(last_lines[-11][-74:-64],
                                                  START_DATE_TIME)

        # Concatenate the lists together
        new_lines = first_lines + last_lines

    # Re-write the lines to a new file
    file = open(html_fname, 'w')
    for line in new_lines:
        file.write(line)
    file.close()


def main():
    """
    Finds suitable Best Data sites for a given site of interest, based on its
    location and elevation. Then creates plots for given Best Data sites, based
    on Best Data and updates a HTML page displaying the plots.
    """
    # Loop through each trial site
    for (trial_site, site_lat, site_lon,
         trial_height, first_dt, last_dt) in zip(TRIAL_SITES, SITE_LATS,
                                                 SITE_LONS, TRIAL_HEIGHTS,
                                                 FIRST_DTS, LAST_DTS):

        # Find most suitable Best Data site
        bd_sites = uf.best_bd_site(DATA_FILE, site_lat, site_lon, trial_height,
                                   RADIUS_LIMIT)

        # Get forecasts based on selection of Best Data sites and make some
        # plots
        get_bd_df(bd_sites, trial_site, first_dt, last_dt)

        # Update html page
        update_html(bd_sites, trial_site, trial_height)


if __name__ == "__main__":
    main()
