# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import time

import datset.amarrays as arr
import datset.ambasic as bas
import datset.amcsv as csv
import datset.dset as dat
import datset.linear as lin
import datset.numset as num
import datset.plot as plo
import datset.learn as lea
import datset.geometry as geo


def unit_tests():
    bas.unit_test()
    arr.unit_test()
    csv.unit_test()
    dat.unit_test()
    num.unit_test()
    lin.unit_test()




def run():
    if bas.expensive_assertions2:
        print("**** WARNING: Expensive Assertions switched on")
    print('*******************************************************************')
    unit_tests()
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hello!')  # Press Ctrl+F8 to toggle the breakpoint.
    quadratic_test()
    quadratic_logistic_test()
    multiclass_misleading_test()

    a = dat.load("banana.csv")
    print('loaded successfully!')
    a.assert_ok()
    a.explain()
    # a.subcols('ascent', 'distance').explain()
    # num.noomset_from_datset(a).explain()

    start_time = time.perf_counter()

    print('\n**********************\n\n')
    output = a.subset('kjoules').discretize('energy', 3)
    assert isinstance(output, dat.Datset)
    #    inputs = a.without(output).without(a.subset('name', 'date'))
    inputs = a.subset('ascent', 'distance')
    assert isinstance(inputs, dat.Datset)

    mc, mod = lin.piclass_glm().train(inputs, output)
    out_name = output.colname(0).string()
    print(
        f"Here's the model CLASS we learned to predict {out_name} from {inputs.colnames().pretty_string()}")
    mc.explain()
    print(
        f"Here's the model we learned to predict {out_name} from {inputs.colnames().pretty_string()}")
    mod.explain()
    plo.show_model(mod, inputs, output)
    check = mc.train(inputs, output)
    print(f'checking retraining the model class gives this:')
    check.explain()

    assert mod.loosely_equals(check)

    n = 250
    d = dat.datset_of_random_unit_floats('a', n).appended_with(dat.datset_of_random_unit_floats('b', n))
    c = arr.strings_empty()

    for aa, b in zip(d.range_floats('a'), d.range_floats('b')):
        dx = aa - 0.5
        dy = b - 0.5
        cls = 'dog'
        if dx * dx + dy * dy < 0.1:
            cls = 'cat'
        elif dx * dy > 0:
            cls = 'pig'
        elif bas.range_random(0.0, 1.0) > 0.75:
            cls = 'cat'

        c.add(cls)

    out = dat.datset_of_strings('c', c)
    print(out.__doc__)

    mc, mod = lin.piclass_glm().train(d, out)
    mc.explain()
    mod.explain()
    plo.show_model(mod, d, out)

    ld = dat.learn_data_from_datsets(a.subset('ascent'), a.subset('kjoules'))
    train, test = ld.train_test_split(0.67)

    mod = lin.model_class_glm(2).train_from_learn_data(train)
    mod.explain()
    assert isinstance(mod, lea.Model)

    d = mod.batch_predict(test.inputs())
    d.explain()

    finish_time = time.perf_counter()

    print(f"That took {finish_time - start_time} seconds")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
