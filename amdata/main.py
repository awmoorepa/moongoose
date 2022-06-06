# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import datset.ambasic as bas
import datset.amarrays as arr
import datset.amcsv as csv
import datset.dset as dat
import datset.numset as num
import datset.learn as lea
import datset.plot as plo


def unit_tests():
    bas.unit_test()
    arr.unit_test()
    csv.unit_test()
    dat.unit_test()
    num.unit_test()


def run():
    if bas.expensive_assertions:
        print("**** WARNING: Expensive Assertions switched on")
    print('*******************************************************************')
    unit_tests()
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hello!')  # Press Ctrl+F8 to toggle the breakpoint.
    a = dat.load("test")
    print('loaded successfully!')
    a.assert_ok()
    a.explain()
    # a.subcols('ascent', 'distance').explain()
    # num.noomset_from_datset(a).explain()

    print('\n**********************\n\n')
    output = a.named_column_from_string('weight')
    inputs = a.without_column(output)
    assert isinstance(inputs, dat.Datset)
    mod = lea.learner_type_linear().train(inputs, output)
    assert isinstance(mod, lea.Model)
    mod.explain()

    for i in range(inputs.num_rows()):
        rw = inputs.row(i)
        assert isinstance(rw, dat.Row)
        print(f'predict {output.colname().string()} for this row: {rw.string()}')
        mod.predict(rw).explain()

    b_inputs = a.subset('hour')
    b_output = a.named_column_from_string('weight')
    mod = lea.learner_type_linear().train(b_inputs,b_output)
    assert isinstance(mod, lea.Model)
    plo.show_model(mod,b_inputs,b_output)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
