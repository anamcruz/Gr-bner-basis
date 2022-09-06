# pylint: disable=consider-using-f-string
# pylint: disable = unrecognized-option
"""Observers, or loggers, that keep track of internal state changes
of subjects being observed."""


class ListAppendObserver:
    """Class of objects to log state of a variable."""

    def __init__(self):
        self._log = []

    def update(self, state):
        self._log.append(state)

    def get_logs(self):
        return self._log


class SilentObserver:
    """Class of object to be used as default argument"""

    def __init__(self):
        pass

    def update(self, unused_state):
        pass
