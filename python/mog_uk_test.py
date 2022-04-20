import sys

import useful_functions as uf


def main(new_data, hall):

if __name__ == "__main__":

    # Print time
    time_1 = uf.print_time('started')

    try:
        new_data = sys.argv[1]
    except:
        print('WARNING! Arguments not set correctly so will exit python '
              'script')
        exit()

    main(new_data, 'xcel01')
