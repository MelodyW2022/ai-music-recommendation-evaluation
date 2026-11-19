import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class TrackRecord:
    """
    Normalized track data from the real Spotify-style CSV dataset.
    """

    track_name: str
    artist_name: str
    album_name: str
    genre: str
    popularity: float
    danceability: float
    energy: float
    acousticness: float
    valence: float
    tempo: float


REQUIRED_COLUMNS = {
    "track_name",
    "artists",
    "album_name",
    "track_genre",
    "popularity",
    "danceability",
    "energy",
    "acousticness",
    "valence",
    "tempo",
}


def load_track_records(csv_path: str, limit: Optional[int] = None) -> List[TrackRecord]:
    """
    Load Spotify-style CSV rows into normalized TrackRecord objects.

    Invalid rows are skipped so one bad record does not break the whole benchmark.
    """
    path = Path(csv_path)
    records: List[TrackRecord] = []

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        _validate_columns(reader.fieldnames)

        for row in reader:
            if limit is not None and len(records) >= limit:
                break

            try:
                records.append(_row_to_track_record(row))
            except (KeyError, ValueError):
                continue

    return records


def _validate_columns(fieldnames: Optional[List[str]]) -> None:
    """
    Make sure the CSV has the columns this project expects.
    """
    if fieldnames is None:
        raise ValueError("CSV file is missing a header row.")

    missing_columns = REQUIRED_COLUMNS.difference(fieldnames)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV file is missing required columns: {missing}")


def _row_to_track_record(row: dict) -> TrackRecord:
    """
    Convert one raw CSV row into a TrackRecord.
    """
    track_name = row["track_name"].strip()
    artist_name = row["artists"].strip()
    album_name = row["album_name"].strip()
    genre = row["track_genre"].strip()

    if not track_name or not artist_name or not genre:
        raise ValueError("Track row is missing required text fields.")

    popularity = _parse_float(row["popularity"])
    danceability = _parse_float(row["danceability"])
    energy = _parse_float(row["energy"])
    acousticness = _parse_float(row["acousticness"])
    valence = _parse_float(row["valence"])
    tempo = _parse_float(row["tempo"])

    return TrackRecord(
        track_name=track_name,
        artist_name=artist_name,
        album_name=album_name,
        genre=genre,
        popularity=popularity,
        danceability=danceability,
        energy=energy,
        acousticness=acousticness,
        valence=valence,
        tempo=tempo,
    )


def _parse_float(value: str) -> float:
    """
    Parse a numeric CSV value.
    """
    return float(value.strip())
