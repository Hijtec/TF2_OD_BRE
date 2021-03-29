import numpy as np


class Button:
    """A storage class for button instance.

    Attributes:
        x_raw:                  X coordinate from button recognition data
        y_raw:                  Y coordinate from button recognition data
        n_raw:                  Proposed number from button recognition data
        n_correct:              An attribute indicating whether the button is suspected to be mislabeled
        col:                    Column coordinate/position
        row:                    Row coordinate/position
    """

    def __init__(self, x_raw, y_raw, n_raw):
        """Initializes the class with position parameters."""
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.n_raw = n_raw
        self.n_correct = None
        self.n_valid = None

        self.col = None
        self.row = None


class Panel:
    """A class for button panel.

    Attributes:
        buttons:                List of Button instances
        rows:                   List of ordered unique rows
        cols:                   List of ordered unique columns
        priority_lr:            Boolean of left-right sequence priority
        priority_vh:            Boolean of horizontal-vertical sequence priority

    Methods:
        assign_buttons:         Assigns coordinates of button objects
    """

    def __init__(self, buttons, rows, cols, priority_lr, priority_vh):
        self.buttons = buttons
        self.rows = rows
        self.cols = cols
        self.priority_lr = priority_lr
        self.priority_vh = priority_vh
        self.assign_buttons()

    def assign_buttons(self):
        """A method for assigning buttons coords to a detection instance."""
        i = 0
        for row in self.rows:
            i += 1
            for item in row:
                self.buttons[item].row = i
        j = 0
        for col in self.cols:
            j += 1
            for item in col:
                self.buttons[item].col = j


class Template:
    """A class for assigning a template to detection instance.

    Attributes:
        buttons_raw                 List of Button classes
        softmax_pred                List of lists of x', x'', x''' prediction labels of buttons
        rows                        List of lists of ordered unique rows
        cols                        List of lists of ordered unique columns

        _n_ranks:                   List of integers representing rank of template parameter candidates
        _n_ranks_row_suppressed:    List of booleans whether a row was suppressed while counting rank

        priority_lr:                Boolean of left-right sequence priority
        priority_vh:                Boolean of horizontal-vertical sequence priority

        seq:                        List of integers representing sequence of raw button labels
        del_seq_index:              Integer of deleted sequence element (during row suppression)
        seq_is_correct:             List of booleans representing whether a wrong button labels
                                    were detected in sequence
        seq_corrected:              List of integers representing sequence of fixed button numbers
        seq_is_correct_corrected:   List of booleans representing whether a wrong button labels
                                    were detected in sequence
        jump_button_detected:       Boolean of jump button presence detection

        buttons_ordered_corrected:  List of Button classes corrected with fixed labels

    Methods:
        assign_template:            Finds best template candidate and sets priority_XX based on their ranks
        flatten_sequence:           Flattens number sequence (suppresses odd rows for priority_vh = False)
        find_seq_error              Finds errors in numbering buttons
        TODO: fix sequence:         Fixes the number sequence
        order buttons:              Creates an ordered list of button objects for further use
    """

    def __init__(self, buttons_raw, softmax_pred, rows_ordered, cols_ordered):
        """Initiates the Template object."""
        self.buttons_raw = buttons_raw
        self.softmax_pred = softmax_pred
        self.rows = rows_ordered
        self.cols = cols_ordered

        # Template assignment ranking
        self._n_ranks = None
        self._n_ranks_row_suppressed = None

        # Template numbering parameters
        self.priority_lr = None
        self.priority_vh = None

        # Template sequencing
        self.seq = None
        self.del_seq_index = None
        self.seq_is_correct = None
        self.seq_corrected = None
        self.seq_is_correct_corrected = None
        self.jump_button_detected = False

        # Button assignment using Template
        self.buttons_ordered_corrected = None

        # Methods used
        self._n_ranks, self._n_ranks_row_suppressed, self.priority_lr, self.priority_vh = self.assign_template_params()
        self.seq, self.del_seq_index = self.flatten_sequence()
        self.seq_is_correct = self.find_seq_error()
        self.seq_corrected, self.seq_is_correct_corrected = self.fix_seq(self.seq)
        self.buttons_ordered_corrected = self.order_buttons()
        self.visualize_buttons()

    def assign_template_params(self):
        """Counts the ranks of possible numbering sequence, assigns counting priority_XX to Template."""
        rank_h_lr, row_suppressed_h_lr = self._count_rank(forwards=True, axis_choice="rows")  # left-to-right by rows
        rank_h_rl, row_suppressed_h_rl = self._count_rank(forwards=False, axis_choice="rows")  # right-to-left by rows
        rank_v_lr, row_suppressed_v_lr = self._count_rank(forwards=True, axis_choice="cols")  # left-to-right by cols
        rank_v_rl, row_suppressed_v_rl = self._count_rank(forwards=False, axis_choice="cols")  # right-to-left by cols

        n_ranks = [rank_h_lr, rank_h_rl, rank_v_lr, rank_v_rl]
        n_ranks_row_suppressed = [row_suppressed_h_lr,
                                  row_suppressed_h_rl,
                                  row_suppressed_v_lr,
                                  row_suppressed_v_rl]

        max_rank_index = np.argmax(np.array(n_ranks))  # gets the index of best candidate
        if max_rank_index == 0 or max_rank_index == 2:
            priority_lr = True
        else:
            priority_lr = False
        if max_rank_index == 0 or max_rank_index == 1:
            priority_vh = True
        else:
            priority_vh = False

        return n_ranks, n_ranks_row_suppressed, priority_lr, priority_vh

    def _count_rank(self, forwards=True, axis_choice="rows"):
        """Ranks a list with chosen order.

        Args:
            forwards: bool depending on if we want to count "forwards" OR "backwards"
            axis_choice: String "rows" OR "cols" depending on whether we rank sequence of rows or columns.

        Returns:
            rank:    Rank of the tested sequence type.
            first_row_suppressed
        """
        # Determining the direction in which we will count rank.
        if forwards:
            direction = 1
        else:
            direction = -1

        # Making an iterable list.
        listed_numbers = []
        first_row_suppressed = False
        if axis_choice == "rows":
            # create an iterable list
            for row in self.rows:
                for i in row:
                    listed_numbers.append(self.buttons_raw[i].n_raw)
        elif axis_choice == "cols":
            rows_suppressed, first_row_suppressed, __ = self._suppress_odd_rows(self.rows)
            if first_row_suppressed:
                cols = self._recalculate_cols(rows_suppressed)
            else:
                cols = self.cols
            # create an iterable list
            for col in cols:
                for i in col:
                    listed_numbers.append(self.buttons_raw[i].n_raw)

        # Defining starting positions to count from.
        if direction == 1:
            curr, foll = 0, 0
        else:
            curr, foll = -1, -1
        rank = 0

        # Iterating through the list.
        for _ in range(len(listed_numbers) - 1):
            foll += direction
            absdiff = abs(listed_numbers[foll]) - abs(listed_numbers[curr])
            if listed_numbers[curr] < listed_numbers[foll] and absdiff < 3:
                rank += 1  # The sequence suits the chosen direction.
            curr = foll
        return rank, first_row_suppressed

    @staticmethod
    def _suppress_odd_rows(rows):
        """Suppresses first row of panel for counting rank if the row is smaller than average of all others."""
        n_buttons = 0
        first_row_suppressed = False
        del_index = None

        for row in rows:
            n_buttons += len(row)
        avg_in_row = n_buttons / len(rows)

        if avg_in_row > len(rows[0]):
            del_index = rows[0][:]
            suppressed_rows = np.delete(rows, del_index)
            first_row_suppressed = True
        else:
            suppressed_rows = rows

        return suppressed_rows, first_row_suppressed, del_index

    @staticmethod
    def _recalculate_cols(suppressed_rows):
        """Recalculates columns after the potential suppression of first row."""
        cols_recalculated_suppressed = []
        n_cols = 0
        for row in suppressed_rows:
            # find the number of columns
            if len(row) > n_cols:
                n_cols = len(row)

        for col_i in range(n_cols):
            recalculated_col = []
            for row in suppressed_rows:
                if len(row) - 1 >= col_i:
                    recalculated_col.append(row[col_i])
            cols_recalculated_suppressed.append(recalculated_col)

        return cols_recalculated_suppressed

    def flatten_sequence(self):
        """Creates an iterable list based on assigned template."""
        seq = []
        del_index = None

        if self.priority_vh:
            if self.priority_lr:
                rows_private = self.rows

            elif not self.priority_lr:
                rows_private = []
                for row in self.rows:
                    rows_private_row = []
                    flip = -1
                    # reverse the order of elements
                    for _ in range(len(row)):
                        rows_private_row.append(row[flip])
                        flip -= 1
                    rows_private.append(rows_private_row)
            else:
                raise ValueError("Priorities have to be boolean")

            for row in rows_private:
                curr = 0
                for _ in row:
                    seq.append(self.buttons_raw[row[curr]].n_raw)
                    curr += 1
        elif not self.priority_vh:
            rows_suppressed, __, del_index = self._suppress_odd_rows(self.rows)
            cols_suppressed = self._recalculate_cols(rows_suppressed)

            if self.priority_lr:
                cols_private = cols_suppressed

            elif not self.priority_lr:
                cols_private = []
                flip = -1
                for __ in cols_suppressed:
                    cols_private.append(cols_suppressed[flip])
                    flip -= 1
            else:
                raise ValueError("Priorities have to be boolean")

            for col in cols_private:
                curr = 0
                for _ in col:
                    seq.append(self.buttons_raw[col[curr]].n_raw)
                    curr += 1
        else:
            raise ValueError("Priorities have to be boolean")

        return seq, del_index

    def find_seq_error(self):
        """Based on template parameters, finds out which numbers are wrong."""
        seq = self.seq
        seq_correct = []
        if seq[0] == seq[-1] - len(seq) + 1 or seq[0] == seq[1] - 1:
            seq_correct.append(True)
        else:
            seq_correct.append(False)

        for i in range(len(seq) - 2):
            if seq[i + 1] == seq[i] + 1 or seq[i + 1] == seq[i + 2] - 1:
                seq_correct.append(True)
            else:
                seq_correct.append(False)

        if seq[-1] == seq[-2] + 1 or seq[-1] == seq[0] + len(seq) - 1:
            seq_correct.append(True)
        else:
            seq_correct.append(False)

        return seq_correct

    def fix_seq(self, seq):
        """Fixes a sequence of buttons."""
        # TODO: fix if rows_suppressed?
        seq_old = np.array(seq)
        seq_is_correct = self.seq_is_correct.copy()
        seq_numbers_corrected, seq_index_corrected = [], []
        for i in range(len(seq) - 1):  # omit first and last button
            seq_index = i
            if not seq_is_correct[seq_index]:
                found_valid_number = False
                valid_number_index = seq_index + 1

                while not found_valid_number:
                    if valid_number_index >= len(seq_is_correct):
                        print("Could not find valid number.")
                        valid_number_index = valid_number_index - 1
                        break
                    if seq_is_correct[valid_number_index]:
                        found_valid_number = True
                    elif not seq_is_correct[valid_number_index]:
                        valid_number_index += 1

                if (abs(seq[seq_index - 1] - seq[valid_number_index])) == (abs(valid_number_index - (seq_index - 1))):
                    # TODO: check this iterator
                    for i in range(seq_index, valid_number_index):
                        number_changed = seq[i]
                        seq[i] = seq[seq_index - 1] + (seq_index - i) + 1
                        seq_numbers_corrected.append(seq[i])
                        seq_index_corrected.append(i)
                        seq_is_correct[seq_index] = True
                        print(f"Replaced index({i}) with number({number_changed}->{seq[i]})")

                elif (abs(seq[seq_index - 1] - seq[valid_number_index])) != (abs(valid_number_index - (seq_index - 1))):
                    print("Jump button detected.")
                    jump_button = True
                    pred = self.softmax_pred
                    for proposal in range(3):
                        if pred[seq_index][proposal] == seq[valid_number_index] + (
                            seq_index - valid_number_index
                        ) or pred[seq_index][proposal] == seq[seq_index] - (
                            (seq_index - 1) - seq_index
                        ):
                            proposal_rank = 1
                        else:
                            proposal_rank = 0
                        if proposal_rank == 1:
                            seq[i] = pred[seq_index][proposal]
                            print(f"Replaced index({i}) with number({seq[i]})")
                            break

                    # TODO:not tested
        print(f"Button labels  {seq_old} \nfixed to array {np.array(seq)}")
        return seq, seq_is_correct

    def order_buttons(self):
        """Grants buttons their columns and rows and proper number."""
        button_list = self.buttons_raw.copy()
        j = 0
        for i in self.buttons_raw:
            i.n_valid = self.seq_is_correct[j]
            i.n_correct = self.seq[j]
            j += 1

        row_index = 0
        for row in self.rows:
            row_index += 1
            for item in row:
                button_list[item].row = row_index

        col_index = 0
        for col in self.cols:
            col_index += 1
            for item in col:
                button_list[item].col = col_index
        return button_list

    def visualize_buttons(self):
        for but in self.buttons_ordered_corrected:
            print(but.__dict__)