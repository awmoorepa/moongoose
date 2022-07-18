from __future__ import annotations

from typing import List, Iterator, Tuple

import matplotlib.pyplot as plt

import datset.dset as dat
import datset.amarrays as arr
import datset.geometry as geo
import datset.ambasic as bas
import datset.numset as noo
import datset.learn as lea
import datset.distribution as dis


def datset_encoding(ds: dat.Datset) -> str:
    assert isinstance(ds, dat.Datset)
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
        return self.floats().range2()

    def add(self, f: float):
        self.m_floats.add(f)

    def min(self) -> float:
        return self.floats().min()

    def max(self) -> float:
        return self.floats().max()

    def tidy_extremes(self) -> Tuple[float, float]:
        return self.floats().tidy_extremes()


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


def floatvec_empty(n_rows: int, label: str) -> Floatvec:
    return floatvec_create(label, arr.floats_empty(n_rows))


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

    def value_to_floats(self, fv: Floatvec) -> arr.FloatsArray:
        v2f = []
        for v in range(self.num_values()):
            v2f.append([])

        for v, f in zip(self.range_values_by_row(), fv.range()):
            v2f[v].append(f)
        return arr.floats_array_from_list_of_lists(v2f)

    def range_values_by_value(self) -> Iterator[int]:
        return self.cats().range_values_by_value()

    def range_valnames_by_value(self) -> Iterator[dat.Valname]:
        return self.cats().range_valnames_by_value()

    def range_values_by_row(self) -> Iterator[int]:
        return self.cats().range_values_by_row()

    def valnames(self) -> dat.Valnames:
        return self.cats().valnames()

    def frequencies_with(self, other: Catvec) -> arr.Imat:
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
    fvs = cv.value_to_floats(fv)
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
    z_to_row_to_x = z.value_to_floats(x)
    z_to_row_to_y = z.value_to_floats(y)
    for xs, ys, label in zip(z_to_row_to_x.range(), z_to_row_to_y.range(), z.names()):
        ax.scatter(xs.list(), ys.list(), label=label)

    ax.set_xlabel(x.label())
    ax.set_ylabel(y.label())
    ax.legend()
    plt.show()


def linear_combination(cls: geo.Colors, fs: arr.Floats) -> geo.Color:
    assert cls.len() == fs.len()
    result = geo.black()
    for colo, f in zip(cls.range(), fs.range2()):
        result = result.plus(colo.times(f))
    return result


def show_model_ffc(mod: lea.Model, x: Floatvec, y: Floatvec, z: Catvec):
    fig, ax = plt.subplots()
    z_to_row_to_x = z.value_to_floats(x)
    z_to_row_to_y = z.value_to_floats(y)
    dd = mod.distribution_description()
    assert dd.distribution_type() == dis.DistributionType.multinomial
    n_model_values = dd.num_values()
    novel_valnames = z.valnames().set_minus()
    cls = geo.color_cycle(z.num_values())
    for xs, ys, label, colo in zip(z_to_row_to_x.range(), z_to_row_to_y.range(), z.names(), cls.range()):
        ax.scatter(xs.list(), ys.list(), label=label, c=[colo.list()])

    xlo, xhi = ax.xaxis.get_data_interval()
    ylo, yhi = ax.yaxis.get_data_interval()

    n_cells_per_edge = 30
    row_to_col_to_rgb = []
    for row in range(n_cells_per_edge):
        y_coord = yhi - (yhi - ylo) * (row + 0.5) / n_cells_per_edge
        col_to_rgb = []
        for col in range(n_cells_per_edge):
            x_coord = xlo + (xhi - xlo) * (col + 0.5) / n_cells_per_edge
            model_inputs = dat.record_from_floats(arr.floats_varargs(x_coord, y_coord))
            model_output = mod.predict_from_record(model_inputs)
            assert isinstance(model_output, dis.Multinomial)
            rgb = linear_combination(cls, model_output.floats())
            col_to_rgb.append(rgb.list())
        row_to_col_to_rgb.append(col_to_rgb)

    ax.imshow(row_to_col_to_rgb, alpha=0.2, extent=(xlo, xhi, ylo, yhi), aspect='auto', interpolation='bilinear')

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


def column_description_encoding(cd: dat.ColumnDescription) -> str:
    if cd.coltype() == dat.Coltype.floats:
        return 'f'
    else:
        return 'c'


def fvc_from_named_column(nc: dat.NamedColumn) -> Floatvec:
    return floatvec_create(nc.colname().string(), nc.floats())


def show_model_ff(mod: lea.Model, x: Floatvec, y: Floatvec):
    fig, ax = plt.subplots()
    ax.scatter(x.list(), y.list())
    ax.set_xlabel(x.label())
    ax.set_ylabel(y.label())
    lo, hi = ax.xaxis.get_data_interval()
    print(f'lo={lo}, hi={hi}')
    xs = arr.floats_from_range(lo, hi, 101)
    print(f'xs = {xs.pretty_string()}')
    ys = arr.floats_empty(xs.len())
    for x in xs.range2():
        d = mod.predict_from_record(dat.record_singleton(dat.atom_from_float(x)))
        assert isinstance(d, dis.Gaussian)
        y = d.mean()
        ys.add(y)

    ax.plot(xs.list(), ys.list())
    plt.show()


def show_model_fc(mod: lea.Model, x: Floatvec, y: Catvec):
    fig, (ax_top, ax_bottom) = plt.subplots(2)
    fvs = y.value_to_floats(x)
    ax_bottom.hist(fvs.list_of_lists(), stacked=True, label=y.names())
    ax_bottom.set_xlabel(x.label())
    ax_bottom.set_ylabel(f'frequency({y.label()})')
    ax_bottom.legend()

    lo, hi = x.tidy_extremes()
    xs = arr.floats_from_range(lo, hi, 101)

    row_to_value_to_prob = arr.floats_array_empty()

    length_of_value_to_prob: int
    dd = mod.distribution_description()
    dt = dd.distribution_type()

    if dt == dis.DistributionType.multinomial:
        length_of_value_to_prob = dd.num_values()
    elif dt == dis.DistributionType.binomial:
        length_of_value_to_prob = 2
    else:
        bas.my_error(f"Can't do this kind of plot for distribution type {dt.string()}")
        length_of_value_to_prob = -77

    for x in xs.range2():
        d = mod.predict_from_record(dat.record_singleton(dat.atom_from_float(x)))

        if dt == dis.DistributionType.multinomial:
            assert isinstance(d, dis.Multinomial)
            value_to_prob = d.floats()
        elif dt == dis.DistributionType.binomial:
            assert isinstance(d, dis.Binomial)
            p = d.theta()
            value_to_prob = arr.floats_varargs(1 - p, p)
        else:
            bas.my_error('bad output column for fc model')
            value_to_prob = arr.floats_default()

        assert value_to_prob.len() == length_of_value_to_prob

        row_to_value_to_prob.add(value_to_prob)

    r2v2p = arr.fmat_from_rows(length_of_value_to_prob, row_to_value_to_prob)

    for row_to_prob, label in zip(r2v2p.range_columns(), y.names()):
        assert isinstance(row_to_prob, arr.Floats)
        assert xs.len() == row_to_prob.len()
        ax_top.plot(xs.list(), row_to_prob.list(), label=label)

    ax_top.set_ylabel(f'P({y.label()})')
    ax_bottom.legend()
    plt.show()


def compute_x_to_y_to_prob(input_valnames: dat.Valnames, mod: lea.Model) -> arr.Fmat:
    x_to_y_to_prob = arr.floats_array_empty()
    dd = mod.distribution_description()
    assert dd.distribution_type() == dis.DistributionType.multinomial

    for vn in input_valnames.range():
        a = dat.atom_from_valname(vn)
        mu = mod.predict_from_record(dat.record_singleton(a))
        assert isinstance(mu, dis.Multinomial)
        x_to_y_to_prob.add(mu.floats())

    return arr.fmat_from_rows(dd.num_values(), x_to_y_to_prob)


def show_model_cc(mod: lea.Model, x: Catvec, y: Catvec):
    fig, (ax_top, ax_bottom) = plt.subplots(2)

    im = x.frequencies_with(y)
    bottom_labels = x.names()

    bottom_heights = arr.ints_all_zero(x.num_values())

    for co, na in zip(im.range_columns(), y.names()):
        ax_bottom.bar(bottom_labels, co.list(), label=na, bottom=bottom_heights.list())
        bottom_heights = bottom_heights.plus(co)
        assert isinstance(bottom_heights, arr.Ints)

    ax_bottom.set_xlabel(x.label())
    ax_bottom.set_ylabel(f'frequency({y.label()})')
    ax_bottom.legend()

    bottom_heights = arr.floats_all_zero(x.num_values())
    fm = compute_x_to_y_to_prob(x.valnames(), mod)
    for y_value in range(y.num_values()):
        heights = fm.column(y_value)
        ax_top.bar(bottom_labels, heights.list(), label=y.valnames().valname(y_value).string(),
                   bottom=bottom_heights.list())
        bottom_heights = bottom_heights.plus(heights)

    ax_top.set_ylabel(f'P({y.label()}|{x.label()})')
    ax_top.legend()
    plt.show()


def show_model(mod: lea.Model, inputs: dat.Datset, output: dat.Datset):
    assert output.num_cols() == 1
    assert output.num_records() == inputs.num_records()
    mod_encoding = datset_encoding(inputs.with_named_column(output.named_column(0)))
    train = inputs.with_named_column(output.named_column(0))
    assert isinstance(train, dat.Datset)
    ds_encoding = datset_encoding(train)
    assert mod_encoding == ds_encoding

    if ds_encoding == 'ff':
        show_model_ff(mod, fvc(inputs, 0), fvc(output, 0))
    elif ds_encoding == 'fc':
        show_model_fc(mod, fvc(inputs, 0), cvc(output, 0))
    elif ds_encoding == 'cc':
        show_model_cc(mod, cvc(inputs, 0), cvc(output, 0))
    elif ds_encoding == 'ffc':
        show_model_ffc(mod, fvc(inputs, 0), fvc(inputs, 1), cvc(output, 0))
    else:
        print(f"Sorry, I can't plot data and model with these characteristics: {ds_encoding}")
