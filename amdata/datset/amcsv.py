import enum

import datset.amarrays as arr
import datset.ambasic as bas


class Cstate(enum.Enum):
    begin_line = 0
    begin_word = 1
    in_string = 2
    in_word = 3
    end_string = 4
    finish = 5
    error = 6
    end_line = 7


def trim_string(s: str) -> str:
    is_started = False
    n_spaces_since_non_space = 0
    result = ""

    for c in s:
        if c == ' ':
            if is_started:
                n_spaces_since_non_space += 1
        else:
            is_started = True
            result = result + bas.n_spaces(n_spaces_since_non_space) + c
            n_spaces_since_non_space = 0

    return result


class CsvReader:
    def __init__(self, state: Cstate, word: str, row: arr.Strings, em: bas.Errmess):
        self.m_cstate = state
        self.m_word = word
        self.m_row = row
        self.m_errmess = em
        self.assert_ok()

    def state(self) -> Cstate:
        return self.m_cstate

    def row(self) -> arr.Strings:
        return self.m_row

    def errmess(self) -> bas.Errmess:
        return self.m_errmess

    def is_finished(self) -> bool:
        return self.state() == Cstate.finish or self.state() == Cstate.error

    def is_end_of_line(self) -> bool:
        return self.is_finished() or self.state() == Cstate.end_line

    def next(self, next_state: Cstate):
        self.m_cstate = next_state
        self.assert_ok()

    def emit_char(self, c: bas.Character, next_state: Cstate):
        self.m_word += c.string()
        self.next(next_state)

    def emit_string(self, s: str, next_state: Cstate):
        self.m_word += s
        self.next(next_state)

    def emit_word(self, next_state: Cstate):
        trimmed = trim_string(self.word())
        self.row().add(trimmed)
        self.m_word = ""
        self.next(next_state)

    def emit_error(self, message: str):
        assert message != ""
        self.m_errmess = bas.errmess_error(message)
        self.m_cstate = Cstate.error

    def assert_ok(self):
        assert isinstance(self.m_cstate, Cstate)
        assert self.errmess().is_error() == (self.state() == Cstate.error)

    def state_is(self, cs: Cstate) -> bool:
        return self.state() == cs

    def word(self) -> str:
        return self.m_word


def csv_reader_start() -> CsvReader:
    return CsvReader(Cstate.begin_line, "", arr.strings_empty(), bas.errmess_ok())


class CsvRowOutput:
    def __init__(self, em: bas.Errmess, row: arr.Strings, file_has_ended: int):
        self.m_errmess = em
        self.m_row = row
        self.m_file_has_ended = file_has_ended
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_errmess, bas.Errmess)
        self.m_errmess.assert_ok()
        assert isinstance(self.m_row, arr.Strings)
        self.m_row.assert_ok()
        assert isinstance(self.m_file_has_ended, bool)

    def is_error(self) -> bool:
        return self.errmess().is_error()

    def errmess(self) -> bas.Errmess:
        return self.m_errmess

    def is_finished_successfully(self) -> bool:
        return self.m_file_has_ended and self.is_ok()

    def is_ok(self) -> bool:
        return self.errmess().is_ok()

    def row(self) -> arr.Strings:
        return self.m_row


def csv_row_output_type_errmess(em: bas.Errmess) -> CsvRowOutput:
    return CsvRowOutput(em, arr.strings_empty(), -77)


def csv_row_output_type_ok(row: arr.Strings, file_has_ended: bool) -> CsvRowOutput:
    return CsvRowOutput(bas.errmess_ok(), row, file_has_ended)


def csv_row_output_file_ended_without_row() -> CsvRowOutput:
    return csv_row_output_type_ok(arr.strings_empty(), True)


def csv_row_output_file_ended_with_row(row: arr.Strings):
    return csv_row_output_type_ok(row, True)


def csv_row_output_ok_and_file_not_ended(row: arr.Strings):
    return csv_row_output_type_ok(row, False)


class StringStream:
    def __init__(self, ss: arr.Strings):
        self.m_strings = ss
        self.m_line_num = 0
        self.m_char_num = 0
        self.m_end_of_file = ss.len() == 0
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_strings, arr.Strings)
        self.m_strings.assert_ok()
        assert isinstance(self.m_line_num, int)
        assert isinstance(self.m_char_num, int)
        assert isinstance(self.m_end_of_file, bool)

        if self.m_end_of_file:
            assert self.m_line_num == self.m_strings.len()
            assert self.m_char_num == 0
        else:
            assert self.m_line_num < self.m_strings.len()
            assert 0 <= self.m_char_num <= len(self.m_strings.string(self.m_line_num))

    def next_character(self) -> tuple[bas.Character, bool]:
        if self.m_end_of_file:
            return bas.character_default(), False
        else:
            s = self.m_strings.string(self.m_line_num)
            if self.m_char_num == len(s):
                self.m_line_num += 1
                self.m_char_num = 0
                if self.m_line_num == self.m_strings.len():
                    self.m_end_of_file = True
                return bas.character_newline(), True
            else:
                result = bas.character_from_string(s, self.m_char_num)
                self.m_char_num += 1
                return result, True

    def is_end_of_file(self) -> bool:
        return self.m_end_of_file


def string_stream_from_strings_array(ss: arr.Strings) -> StringStream:
    return StringStream(ss)


class NextRowStatus(enum.Enum):
    maybe_more_lines_to_come = 0
    file_ended_without_extra_row = 1
    file_ended_with_extra_row = 2
    error = 4


def next_csv_row_output(ss: StringStream) -> tuple[arr.Strings, bas.Errmess, NextRowStatus]:
    cs = csv_reader_start()

    while not cs.is_end_of_line():
        c, ok = ss.next_character()
        finished = not ok

        if cs.state_is(Cstate.begin_line):
            if finished:
                cs.next(Cstate.finish)
            elif c.is_char(','):
                cs.emit_word(Cstate.begin_word)
            elif c.is_char('"'):
                cs.next(Cstate.in_string)
            elif c.is_char('\n'):
                cs.next(Cstate.begin_line)
            else:
                cs.emit_char(c, Cstate.in_word)

        elif cs.state_is(Cstate.begin_word):
            if finished:
                cs.emit_word(Cstate.finish)
            elif c.is_char(','):
                cs.emit_word(Cstate.begin_word)
            elif c.is_char('"'):
                cs.next(Cstate.in_string)
            elif c.is_char('\n'):
                cs.emit_word(Cstate.end_line)
            else:
                cs.emit_char(c, Cstate.in_word)

        elif cs.state_is(Cstate.in_string):
            if finished:
                cs.emit_error("File ended in the middle of a quoted string")
            elif c.is_char('\n'):
                cs.emit_string("<return>", Cstate.in_string)
            elif c.is_char('"'):
                cs.next(Cstate.end_string)
            else:
                cs.emit_char(c, Cstate.in_string)

        elif cs.state_is(Cstate.end_string):
            if finished:
                cs.emit_word(Cstate.finish)
            elif c.is_char(','):
                cs.emit_word(Cstate.begin_word)
            elif c.is_char('"'):
                cs.emit_char(c, Cstate.in_string)
            elif c.is_char('\n'):
                cs.emit_word(Cstate.end_line)
            else:
                cs.emit_error(
                    "After you close a quoted string I expect to see a comma, another quote, newline or end of file")

        elif cs.state_is(Cstate.in_word):
            if finished:
                cs.emit_word(Cstate.finish)
            elif c.is_char(','):
                cs.emit_word(Cstate.begin_word)
            elif c.is_char('"'):
                cs.emit_error("I don't expect to find a quote symbol inside an unquoted string")
            elif c.is_char('\n'):
                cs.emit_word(Cstate.end_line)
            else:
                cs.emit_char(c, Cstate.in_word)

        elif cs.state_is(Cstate.finish):
            bas.my_error("We can't find ourselves here")
        elif cs.state_is(Cstate.end_line):
            bas.my_error("We can't find ourselves here")
        elif cs.state_is(Cstate.error):
            bas.my_error("We can't find ourselves here")
        else:
            bas.my_error("Bad CSV state")

    if cs.state_is(Cstate.error):
        return arr.strings_empty(), cs.errmess(), NextRowStatus.error
    elif cs.state_is(Cstate.finish):
        if cs.row().len() == 0:
            return arr.strings_empty(), bas.errmess_ok(), NextRowStatus.file_ended_without_extra_row
        else:
            return cs.row(), bas.errmess_ok(), NextRowStatus.file_ended_with_extra_row
    else:
        assert cs.state_is(Cstate.end_line)
        return cs.row(), bas.errmess_ok(), NextRowStatus.maybe_more_lines_to_come


def strings_array_from_string_stream_csv(sts: StringStream) -> tuple[arr.StringsArray, bas.Errmess]:
    result = arr.strings_array_empty()

    while True:
        row, em, nrs = next_csv_row_output(sts)

        if nrs == NextRowStatus.error:
            assert em.is_error()
            return arr.strings_array_empty(), em
        elif nrs == NextRowStatus.maybe_more_lines_to_come:
            assert em.is_ok()
            result.add(row)
        elif nrs == NextRowStatus.file_ended_with_extra_row:
            assert em.is_ok()
            result.add(row)
            return result, bas.errmess_ok()
        else:
            assert nrs == NextRowStatus.file_ended_without_extra_row
            assert em.is_ok()
            return result, bas.errmess_ok()


def strings_array_from_strings_csv(ss: arr.Strings) -> tuple[arr.StringsArray, bas.Errmess]:
    sts = string_stream_from_strings_array(ss)
    return strings_array_from_string_stream_csv(sts)


def csv_test_ok(s: str, r: int, c: int, check: str):
    print(f'csv_test_ok(s,r={r},c={c},check)')
    print(f's = {s}')
    print(f'check = {check}')

    ss = arr.strings_from_lines_in_string(s)

    print(f"***\nss = \n{ss.pretty_string()}***")

    ssa, em = strings_array_from_strings_csv(ss)

    print(f'em = {em.string()}')
    assert em.is_ok()

    print(f'ssa = \n{ssa.pretty_string()}')
    assert ssa.string(r, c) == check


def csv_test_bad(s: str):
    print('csv_test_bad(s)')
    print(f's = \n{s}')
    ss = arr.strings_from_lines_in_string(s)

    print(f'ss = \n{ss.pretty_string()}')

    ssa, em = strings_array_from_strings_csv(ss)
    assert em.is_error()


def unit_test():
    s = "andrew,3,plop\nmary,5,plop"
    csv_test_ok(s, 0, 0, "andrew")
    csv_test_ok(s, 1, 2, "plop")
    s = "\n" + s + "\n"
    csv_test_ok(s, 0, 0, "andrew")
    csv_test_ok(s, 1, 2, "plop")

    s = "  andrew ,3,plop\nmary,,plop\n"
    csv_test_ok(s, 0, 0, "andrew")
    csv_test_ok(s, 1, 1, "")

    s = "  andrew \nmary\n\nlucy\n    william \n"
    csv_test_ok(s, 0, 0, "andrew")
    csv_test_ok(s, 3, 0, "william")

    s = '  andrew,"mary\nlucy",    william \na,b,c\n'
    csv_test_ok(s, 0, 2, "william")
    csv_test_ok(s, 0, 1, "mary<return>lucy")

    s = 'hello,"ther"e,old,friend\n'
    csv_test_bad(s)

    s = 'hello,there\nold\n'  # Note that this would fail the datset reader because different line lengths
    csv_test_ok(s, 0, 1, "there")
    csv_test_ok(s, 1, 0, "old")
