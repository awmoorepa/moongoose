from typing import List, Iterator

import matplotlib.pyplot as plt

import datset.dset as dat
import datset.amarrays as arr
import datset.ambasic as bas
import datset.geometry as geo
import datset.numset as noo
import datset.learn as lea


class Axis:
    def __init__(self, label: str, limits: bas.Interval):
        self.m_label = label
        self.m_limits = limits
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_label, str)
        assert isinstance(self.m_limits, bas.Interval)
        self.m_limits.assert_ok()
        assert self.m_limits.width() > 0

    def interval(self) -> bas.Interval:
        return self.m_limits

    def label(self) -> str:
        return self.m_label


def set_plt_limits(is_x_axis: bool, plt_axis, limits: bas.Interval):
    lo = limits.lo()
    hi = limits.hi()
    if is_x_axis:
        plt_axis.set_xlim(lo, hi)
    else:
        plt_axis.set_ylim(lo, hi)


def set_plt_label(is_x_axis: bool, plt_axis, label: str):
    if is_x_axis:
        plt_axis.set_xlabel(label)
    else:
        plt_axis.set_ylabel(label)


def set_plt_axis(is_x_axis: bool, plt_axis, am_axis: Axis):
    set_plt_limits(is_x_axis, plt_axis, am_axis.interval())
    set_plt_label(is_x_axis, plt_axis, am_axis.label())


def set_plt_x_axis(plt_axis, am_axis: Axis):
    set_plt_axis(True, plt_axis, am_axis)


def set_plt_y_axis(plt_axis, am_axis: Axis):
    set_plt_axis(False, plt_axis, am_axis)


class Diagram:
    def __init__(self, x: Axis, y: Axis, vs: geo.Vecs):
        self.m_x_axis = x
        self.m_y_axis = y
        self.m_data = vs
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_x_axis, Axis)
        self.m_x_axis.assert_ok()
        assert isinstance(self.m_y_axis, Axis)
        self.m_y_axis.assert_ok()
        assert isinstance(self.m_data, geo.Vecs)
        self.m_data.assert_ok()
        assert self.data_border().loosely_contains_vecs(self.vecs())

    def data_border(self) -> geo.Rect:
        return geo.rect_from_intervals(self.x_interval(), self.y_interval())

    def vecs(self) -> geo.Vecs:
        return self.m_data

    def x_interval(self) -> bas.Interval:
        return self.x_axis().interval()

    def x_axis(self) -> Axis:
        return self.m_x_axis

    def y_interval(self) -> bas.Interval:
        return self.y_axis().interval()

    def y_axis(self) -> Axis:
        return self.m_y_axis

    def show(self):
        fig, ax = plt.subplots()
        x, y = self.vecs().unzip()
        xli = x.list()
        yli = y.list()
        ax.scatter(xli, yli)
        set_plt_x_axis(ax, self.x_axis())
        set_plt_y_axis(ax, self.y_axis())
        plt.show()


class NamedFloats:
    def __init__(self, name: str, fs: arr.Floats):
        self.m_name = name
        self.m_floats = fs
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_name, str)
        assert isinstance(self.m_floats, arr.Floats)
        self.m_floats.assert_ok()

    def axis(self) -> Axis:
        return axis_create(self.name(), self.floats().tidy_surrounder())

    def floats(self) -> arr.Floats:
        return self.m_floats

    def name(self) -> str:
        return self.m_name


def named_floats_create(name: str, fs: arr.Floats) -> NamedFloats:
    return NamedFloats(name, fs)


def named_floats_from_datset(ds: dat.Datset, col: int) -> NamedFloats:
    assert ds.column_is_floats(col)
    return named_floats_create(ds.colname(col).string(), ds.floats(col))


def diagram_create(x: Axis, y: Axis, vs: geo.Vecs) -> Diagram:
    return Diagram(x, y, vs)


def diagram_bivariate_ff(x: NamedFloats, y: NamedFloats) -> Diagram:
    return diagram_create(x.axis(), y.axis(), geo.vecs_from_floats(x.floats(), y.floats()))


def axis_create(name: str, limits: bas.Interval) -> Axis:
    return Axis(name, limits)


def axis_default():
    return axis_create('default', bas.interval_unit())


def diagram_default():
    return diagram_create(axis_default(), axis_default(), geo.vecs_empty())


def datset_encoding(ds: dat.Datset) -> str:
    assert isinstance(ds,dat.Datset)
    result = ""
    for c in ds.range_columns():
        result += 'f' if c.is_floats() else 'c'
    return result


class Floatvec:
    def __init__(self, label: str, fs: arr.Floats):
        self.m_label = label
        self.m_floats = fs
        self.assert_ok()

    def list(self) -> List[float]:
        return self.floats().list()

    def floats(self) -> arr.Floats:
        return self.m_floats

    def label(self) -> str:
        return self.m_label

    def assert_ok(self):
        assert isinstance(self.m_label, str)
        assert isinstance(self.m_floats, arr.Floats)
        self.m_floats.assert_ok()

    def range(self) -> Iterator[float]:
        return self.floats().range()

    def add(self, f: float):
        self.m_floats.add(f)


def floatvec_create(label: str, fs: arr.Floats):
    return Floatvec(label, fs)


def floatvec_from_named_column(nc: dat.NamedColumn) -> Floatvec:
    return floatvec_create(nc.colname().string(), nc.floats())


class Floatvecs:
    def __init__(self, float_label: str, cat_label: str, li: List[Floatvec]):
        self.m_float_label = float_label
        self.m_cat_label = cat_label
        self.m_list = li
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_float_label, str)
        assert isinstance(self.m_cat_label, str)
        assert isinstance(self.m_list, list)
        for fv in self.m_list:
            assert isinstance(fv, Floatvec)

    def list_of_lists(self) -> List[List[float]]:
        result = []
        for fv in self.range():
            result.append(fv.list())
        return result

    def add(self, fv: Floatvec):
        self.m_list.append(fv)

    def range(self) -> Iterator[Floatvec]:
        for fv in self.m_list:
            yield fv

    def floatvec(self, v: int) -> Floatvec:
        assert 0 <= v < self.len()
        return self.m_list[v]

    def len(self) -> int:
        return len(self.m_list)


def floatvecs_empty(float_label: str, cat_label: str):
    return Floatvecs(float_label, cat_label, [])


def floatvec_empty(label: str) -> Floatvec:
    return floatvec_create(label, arr.floats_empty())


# Catvec is not a simple analog of dat.Cats because it treats Bools just like Cats (with Valnames True and False etc.)
class Catvec:
    def __init__(self, label: str, cs: dat.Cats):
        self.m_label = label
        self.m_cats = cs
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_label, str)
        assert isinstance(self.m_cats, dat.Cats)
        self.m_cats.assert_ok()

    def names(self) -> List[str]:
        return self.cats().valnames().strings().list()

    def cats(self) -> dat.Cats:
        return self.m_cats

    def frequencies(self) -> List[int]:
        h = self.cats().values().histogram()
        assert isinstance(h, arr.Ints)
        return h.list()

    def label(self) -> str:
        return self.m_label

    def floats_array(self, fv: Floatvec) -> arr.FloatsArray:
        result = arr.floats_array_of_empties(self.num_values())
        for v, f in zip(self.range_values_by_row(), fv.range()):
            result.floats(v).add(f)
        return result

    def range_values_by_value(self) -> Iterator[int]:
        return self.cats().range_values_by_value()

    def range_valnames_by_value(self) -> Iterator[dat.Valname]:
        return self.cats().range_valnames_by_value()

    def range_values_by_row(self) -> Iterator[int]:
        return self.cats().range_values_by_row()

    def valnames(self) -> dat.Valnames:
        return self.cats().valnames()

    def frequencies_with(self,
                         other) -> arr.Imat:  # other is type catvec. result[i,j] = count how often self is value i and other is value j
        assert isinstance(other, Catvec)
        result = arr.row_indexed_imat_of_zeroes(self.num_values(), other.num_values())

        for sv, ov in zip(self.range_values_by_row(), other.range_values_by_row()):
            result.increment_by_one(sv, ov)

        return arr.imat_create(result)

    def num_values(self) -> int:
        return self.cats().num_values()


def show_c(cv: Catvec):
    fig, ax = plt.subplots()
    ax.bar(cv.names(), cv.frequencies())
    ax.set_xlabel(cv.label())
    plt.show()


def catvec_create(label: str, c: dat.Cats):
    return Catvec(label, c)


def cats_from_bools(bs: arr.Bools) -> dat.Cats:
    value_for_false = 0
    string_for_false = bas.string_from_bool(False)
    value_for_true = 1
    string_for_true = bas.string_from_bool(True)
    ss = arr.strings_empty()
    ss.add(string_for_false)
    ss.add(string_for_true)
    vns = dat.valnames_from_strings(ss)

    row_to_value = arr.ints_empty()
    for b in bs.range():
        value = value_for_true if b else value_for_false
        row_to_value.add(value)
    return dat.cats_create(row_to_value, vns)


def catvec_from_named_column(nc: dat.NamedColumn) -> Catvec:
    cs = nc.cats() if nc.is_cats() else cats_from_bools(nc.bools())
    return catvec_create(nc.colname().string(), cs)


def cvc(ds: dat.Datset, col: int) -> Catvec:
    return catvec_from_named_column(ds.named_column(col))


def fvc(ds: dat.Datset, col: int) -> Floatvec:
    return floatvec_from_named_column(ds.named_column(col))


def show_f(fv: Floatvec):
    fig, ax = plt.subplots()
    ax.hist(fv.list())
    ax.set_xlabel(fv.label())
    plt.show()


def show_ff(x: Floatvec, y: Floatvec):
    fig, ax = plt.subplots()
    ax.scatter(x.list(), y.list())
    ax.set_xlabel(x.label())
    ax.set_ylabel(y.label())
    plt.show()


def show_fc(fv: Floatvec, cv: Catvec):
    fig, ax = plt.subplots()
    fvs = cv.floats_array(fv)
    ax.hist(fvs.list_of_lists(), stacked=True, label=cv.names())
    ax.set_xlabel(fv.label())
    ax.set_ylabel(f'frequency({cv.label()})')
    ax.legend()
    plt.show()


def show_cc(x: Catvec, y: Catvec):
    fig, ax = plt.subplots()
    im = x.frequencies_with(y)
    bottom_labels = x.names()

    bottom_heights = arr.ints_all_zero(x.num_values())

    print(f'bottom_labels = {x.valnames().strings().pretty_string()}')
    print(f'im = \n{im.pretty_string()}')
    for co, na in zip(im.range_columns(), y.names()):
        print(f'co = {co.pretty_string()}')
        print(f'na = {na}')
        print(f'co.list() = {co.list()}')
        ax.bar(bottom_labels, co.list(), label=na, bottom=bottom_heights.list())
        bottom_heights = bottom_heights.plus(co)
        assert isinstance(bottom_heights, arr.Ints)

    ax.set_xlabel(x.label())
    ax.set_ylabel(f'frequency({y.label()})')
    ax.legend()
    plt.show()


def show_ffc(x: Floatvec, y: Floatvec, z: Catvec):
    fig, ax = plt.subplots()
    z_to_row_to_x = z.floats_array(x)
    z_to_row_to_y = z.floats_array(y)
    for xs, ys, label in zip(z_to_row_to_x.range(), z_to_row_to_y.range(), z.names()):
        ax.scatter(xs.list(), ys.list(), label=label)

    ax.set_xlabel(x.label())
    ax.set_ylabel(y.label())
    ax.legend()
    plt.show()


def show(ds: dat.Datset):
    en = datset_encoding(ds)
    if en == "c":
        show_c(cvc(ds, 0))
    elif en == 'f':
        show_f(fvc(ds, 0))
    elif en == 'ff':
        show_ff(fvc(ds, 0), fvc(ds, 1))
    elif en == 'fc':
        show_fc(fvc(ds, 0), cvc(ds, 1))
    elif en == 'cf':
        show_fc(fvc(ds, 1), cvc(ds, 0))
    elif en == 'cc':
        show_cc(cvc(ds, 0), cvc(ds, 1))
    elif en == 'ffc':
        show_ffc(fvc(ds, 0), fvc(ds, 1), cvc(ds, 2))
    elif en == 'fcf':
        show_ffc(fvc(ds, 0), fvc(ds, 2), cvc(ds, 1))
    elif en == 'cff':
        show_ffc(fvc(ds, 1), fvc(ds, 2), cvc(ds, 0))
    else:
        print(f"Sorry. I can't draw a plot for a datset with these characteristics: {en}")


def transformer_encoding(tf: noo.Transformer) -> str:
    if tf.is_float():
        return 'f'
    else:
        return 'c'


def transformers_encoding(tfs: noo.Transformers) -> str:
    result = ''
    for tf in tfs.range():
        result += transformer_encoding(tf)
    return result


def learner_encoding(le: lea.Learner) -> str:
    if le.learn_type() == lea.Learntype.linear:
        return 'f'
    else:
        return 'c'


def model_encoding(mod: lea.Model) -> str:
    return transformers_encoding(mod.transformers()) + learner_encoding(mod.learner())


def fvc_from_named_column(nc: dat.NamedColumn) -> Floatvec:
    return floatvec_create(nc.colname().string(), nc.floats())


def show_model_ff(mod: lea.Model, x: Floatvec, y: Floatvec):
    fig, ax = plt.subplots()
    ax.scatter(x.list(), y.list())
    ax.set_xlabel(x.label())
    ax.set_ylabel(y.label())
    lo, hi = ax.xlim()
    xs = arr.floats_from_range(lo, hi, 101)
    ys = mod.predict_floats(xs)
    ax.plot(xs.list(), ys.list())
    plt.show()


def show_model(mod: lea.Model, inputs: dat.Datset, output: dat.NamedColumn):
    mod_encoding = model_encoding(mod)
    train = inputs.with_named_column(output)
    assert isinstance(train,dat.Datset)
    ds_encoding = datset_encoding(train)
    assert mod_encoding == ds_encoding

    if ds_encoding == 'ff':
        show_model_ff(mod, fvc(inputs, 0), fvc_from_named_column(output))
    else:
        print(f"Sorry, I can't plot data and model with these characteristics: {ds_encoding}")
