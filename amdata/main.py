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
import datset.linear as lin


def unit_tests():
    bas.unit_test()
    arr.unit_test()
    csv.unit_test()
    dat.unit_test()
    num.unit_test()
    lin.unit_test()


def quadratic_test():
    ds = dat.datset_of_random_unit_floats('a', 100)
    ds.define_column('b', '*', 'a', 'a')
    ds.explain()
    mod = lin.model_class_glm(2).train(ds.subset('a'), ds.subset('b'))
    mod.explain()
    plo.show_model(mod, ds.subset('a'), ds.subset('b'))


def quadratic_logistic_test():
    ds = dat.datset_of_random_unit_floats('a', 30)
    bs = arr.bools_empty()
    for a in ds.range_floats('a'):
        bs.add(0.25 < a < 0.65)
    ds.add_bools_column('b', bs)
    ds.explain()
    mod = lin.model_class_glm(2).train(ds.subset('a'), ds.subset('b'))
    mod.explain()
    plo.show_model(mod, ds.subset('a'), ds.subset('b'))


def test_q3():
    n_records = 40
    ds = dat.datset_of_random_unit_floats('x', n_records).appended_with(
        dat.datset_of_random_unit_floats('y', n_records))
    ss = arr.strings_empty()

    for x, y in zip(ds.range_floats('x'), ds.range_floats('y')):
        dx = x - 0.5
        dy = y - 0.5
        c: str
        if dx * dx + dy * dy < 0.25 * 0.25:
            c = 'middle'
        elif dx > 0 and dy > 0:
            c = 'main'
        else:
            c = 'minor'
        ss.add(c)

    cs = dat.cats_from_strings(ss)
    col = dat.column_from_cats(cs)
    output = dat.datset_from_single_column(dat.colname_create('class'), col)

    mod = lin.model_class_glm(1).train(ds, output)
    plo.show_model(mod, ds, output)
    mod.explain()


def run():
    if bas.expensive_assertions:
        print("**** WARNING: Expensive Assertions switched on")
    print('*******************************************************************')
    unit_tests()
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hello!')  # Press Ctrl+F8 to toggle the breakpoint.
    quadratic_test()
    quadratic_logistic_test()
    a = dat.load("banana.csv")
    print('loaded successfully!')
    a.assert_ok()
    a.explain()
    # a.subcols('ascent', 'distance').explain()
    # num.noomset_from_datset(a).explain()

    print('\n**********************\n\n')
    output = a.subset('kjoules').discretize('energy', 3)
    assert isinstance(output, dat.Datset)
    #    inputs = a.without(output).without(a.subset('name', 'date'))
    inputs = a.subset('ascent')
    assert isinstance(inputs, dat.Datset)
    mod = lin.model_class_glm(1).train(inputs, output)
    assert isinstance(mod, lea.Model)
    mod.explain()

    for i in range(inputs.num_records()):
        rw = inputs.record(i)
        assert isinstance(rw, dat.Record)
        print(f'predict {output.colname(0).string()} for this row: {rw.string()}')
        mod.predict_from_record(rw).explain()

    b_inputs = a.subset('ascent', 'distance')
    mod = lin.model_class_glm(2).train(b_inputs, output)
    assert isinstance(mod, lea.Model)
    plo.show_model(mod, b_inputs, output)

    c = a.subset('ascent').appended_with(output)
    assert isinstance(c, dat.Datset)
    c_train, c_test = c.split(0.6)

    c_train_in = c_train.subset('ascent')
    c_train_out = c_train.subset('energy')
    c_test_in = c_test.subset('ascent')

    mod = lin.model_class_glm(1).train(c_train_in, c_train_out)
    assert isinstance(mod, lea.Model)

    d = mod.batch_predict(c_test_in)
    d.explain()

    #  test_q3()


#    plo.show_model(mod, c_test_in, c_test_out)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
