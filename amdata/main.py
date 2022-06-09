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
    a = dat.load("cycle.csv")
    print('loaded successfully!')
    a.assert_ok()
    a.explain()
    # a.subcols('ascent', 'distance').explain()
    # num.noomset_from_datset(a).explain()

    print('\n**********************\n\n')
    output = a.subset('energy')
    assert isinstance(output, dat.Datset)
    inputs = a.without(output).without(a.subset('name', 'date'))
    assert isinstance(inputs, dat.Datset)
    mod = lea.learner_type_multinomial().train(inputs, output)
    assert isinstance(mod, lea.Model)
    mod.explain()

    for i in range(inputs.num_rows()):
        rw = inputs.row(i)
        assert isinstance(rw, dat.Row)
        print(f'predict {output.colname(0).string()} for this row: {rw.string()}')
        mod.predict(rw).explain()

    b_inputs = a.subset('ascent')
    b_output = a.subset('energy')
    assert isinstance(b_output, dat.Datset)
    mod = lea.learner_type_multinomial().train(b_inputs, b_output)
    assert isinstance(mod, lea.Model)
    plo.show_model(mod, b_inputs, b_output)

    c = a.subset('ascent', 'energy')
    assert isinstance(c, dat.Datset)
    c_train, c_test = c.split(0.6)

    c_train_in = c_train.subset('ascent')
    c_train_out = c_train.subset('energy')
    c_test_in = c_test.subset('ascent')
    c_test_out = c_test.subset('energy')

    mod = lea.learner_type_multinomial().train(c_train_in, c_train_out)
    assert isinstance(mod, lea.Model)

    d = mod.batch_predict(c_test)
    d.explain()

#    plo.show_model(mod, c_test_in, c_test_out)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
