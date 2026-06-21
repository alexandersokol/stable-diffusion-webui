import os
import threading


sequence_number_cache = {}
sequence_number_cache_lock = threading.RLock()


def _sequence_cache_key(path, basename):
    return os.path.abspath(path), basename


def _filename_for_sequence(path, basename, file_decoration, extension, sequence_number):
    fn = f"{sequence_number:05}" if basename == '' else f"{basename}-{sequence_number:04}"
    return os.path.join(path, f"{fn}{file_decoration}.{extension}")


def _find_next_available_filename(path, basename, file_decoration, extension, start_sequence, max_attempts=500):
    for i in range(max_attempts):
        sequence_number = start_sequence + i
        fullfn = _filename_for_sequence(path, basename, file_decoration, extension, sequence_number)
        if not os.path.exists(fullfn):
            return fullfn, sequence_number

    raise FileExistsError(f"Could not find an available filename after {max_attempts} attempts in {path}")


def scan_next_sequence_number(path, basename):
    """
    Determines and returns the next sequence number to use when saving an image in the specified directory.

    The sequence starts at 0.
    """
    result = -1
    basename_prefix = f"{basename}-" if basename != '' else ''
    prefix_length = len(basename_prefix)

    for filename in os.listdir(path):
        if filename.startswith(basename_prefix):
            parts = os.path.splitext(filename[prefix_length:])[0].split('-')
            try:
                result = max(int(parts[0]), result)
            except ValueError:
                pass

    return result + 1


def reset_sequence_number_cache():
    with sequence_number_cache_lock:
        sequence_number_cache.clear()


def get_next_available_filename(path, basename, file_decoration, extension, start_sequence, max_attempts=500):
    fullfn, _ = _find_next_available_filename(path, basename, file_decoration, extension, start_sequence, max_attempts=max_attempts)
    return fullfn


def get_next_sequence_filename(path, basename, file_decoration, extension, max_attempts=500):
    key = _sequence_cache_key(path, basename)

    with sequence_number_cache_lock:
        start_sequence = sequence_number_cache.get(key)
        if start_sequence is None:
            start_sequence = scan_next_sequence_number(path, basename)

        try:
            fullfn, sequence_number = _find_next_available_filename(path, basename, file_decoration, extension, start_sequence, max_attempts=max_attempts)
        except FileExistsError:
            start_sequence = scan_next_sequence_number(path, basename)
            fullfn, sequence_number = _find_next_available_filename(path, basename, file_decoration, extension, start_sequence, max_attempts=max_attempts)

        sequence_number_cache[key] = sequence_number + 1
        return fullfn
