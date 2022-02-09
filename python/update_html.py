import os

# HTML_DIR = '/home/h04/alanyon/public_html/uav'
# URL_START = 'https://www-nwp/~alanyon/uav/html'
# SIDEBAR = '/home/h04/alanyon/public_html/sidebar.shtml'
HTML_DIR = '/home/h05/avapps/public_html/uav'
URL_START = 'https://www-nwp/~avapps/uav/html'
SIDEBAR = '/home/h05/avapps/public_html/uav/sidebar.shtml'


def main(date, site_height, site_name, site_fname):
    """
    Updates html file.
    """
    # File name of html file
    html_fname = f'{HTML_DIR}/html/{site_fname}_mog_uk_fcasts.shtml'

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
        lines[34] = lines[34].replace('TRIAL', site_fname)
        lines[48] = lines[48].replace('NAME', site_name)
        lines[48] = lines[48].replace('HEIGHT', str(site_height))
        lines[76] = lines[76].replace('DATE', date)
        lines[79] = lines[79].replace('TRIAL', site_fname)
        lines[79] = lines[79].replace('NAME', site_name)
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
                           f'{site_name} MOGREPS-UK Forecasts</a></li>\n')

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
        first_lines = lines[:-22]
        last_lines = lines[-22:]

        # Edit html file and append/edit the required lines
        first_lines[-1] = first_lines[-1].replace(' selected="selected"', '')
        first_lines.append('                        <option selected='
                           f'"selected" value="{date}">{date}</option>\n')
        last_lines[-11] = last_lines[-11].replace(last_lines[-11][-82:-71],
                                                  date)

        # Concatenate the lists together
        new_lines = first_lines + last_lines

    # Re-write the lines to a new file
    file = open(html_fname, 'w')
    for line in new_lines:
        file.write(line)
    file.close()


if __name__ == "__main__":
    main('2022020909Z', 77, 'National Grid', 'National_Grid'):
