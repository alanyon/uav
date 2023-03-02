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


HTML_DIR = '/home/h04/alanyon/public_html/uav'
# Change these bits for new trial site/date
# Dates of start and end of trial
FIRST_DT = datetime(2022, 5, 3, 0)  # Year, month, day, hour
LAST_DT = datetime(2022, 5, 5, 1)  # Year, month, day, hour
# Location/height/name of site
LAT = 53.225556
LON = -0.881389
SITE_HEIGHT = 43
SITE_NAME = 'National Rail'


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
        first_lines = lines[:-18]
        last_lines = lines[-18:]

        # Edit html file and append/edit the required lines
        first_lines[-1] = first_lines[-1].replace(' selected="selected"', '')
        first_lines.append('                        <option selected='
                           f'"selected" value="{date}">{date}</option>\n')
        last_lines[-7] = last_lines[-7].replace(last_lines[-7][-82:-71], date)

        # Concatenate the lists together
        new_lines = first_lines + last_lines

    # Re-write the lines to a new file
    file = open(html_fname, 'w')
    for line in new_lines:
        file.write(line)
    file.close()

update_html('2022042718')
