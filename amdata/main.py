# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import datset.amarrays as arr
import datset.amcsv as csv
import datset.dset as dat


def unit_tests():
    arr.unit_test()
    csv.unit_test()
    dat.unit_test()


def run():
    print('*******************************************************************')
    unit_tests()
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hello!')  # Press Ctrl+F8 to toggle the breakpoint.
    a = dat.load("test")
    print('loaded successfully!')
    a.assert_ok()
    a.explain()
    # a.subcols('ascent', 'distance').explain()
    ns.numset_from_datset(a).explain()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
