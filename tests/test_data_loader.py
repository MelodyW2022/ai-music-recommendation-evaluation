import pytest

from src.data_loader import load_track_records
from src.main import load_demo_records


def test_load_track_records_normalizes_spotify_style_csv(tmp_path):
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text(
        "track_name,artists,album_name,track_genre,popularity,"
        "danceability,energy,acousticness,valence,tempo\n"
        "Sample Track,Sample Artist,Sample Album,pop,71,"
        "0.77,0.82,0.18,0.72,136\n",
        encoding="utf-8",
    )

    records = load_track_records(str(csv_path))

    assert len(records) == 1
    assert records[0].track_name == "Sample Track"
    assert records[0].artist_name == "Sample Artist"
    assert records[0].genre == "pop"
    assert records[0].energy == 0.82
    assert records[0].acousticness == 0.18
    assert records[0].valence == 0.72
    assert records[0].danceability == 0.77
    assert records[0].tempo == 136


def test_load_track_records_reports_missing_columns(tmp_path):
    csv_path = tmp_path / "bad_tracks.csv"
    csv_path.write_text(
        "track_name,artists\n"
        "Sample Track,Sample Artist\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required columns"):
        load_track_records(str(csv_path))


def test_load_demo_records_uses_csv_when_available(tmp_path):
    csv_path = tmp_path / "tracks.csv"
    csv_path.write_text(
        "track_name,artists,album_name,track_genre,popularity,"
        "danceability,energy,acousticness,valence,tempo\n"
        "Sample Track,Sample Artist,Sample Album,pop,71,"
        "0.77,0.82,0.18,0.72,136\n",
        encoding="utf-8",
    )

    records = load_demo_records(csv_path=str(csv_path), limit=None)

    assert len(records) == 1
    assert records[0].track_name == "Sample Track"


def test_load_demo_records_falls_back_when_csv_is_missing(tmp_path):
    missing_path = tmp_path / "missing.csv"

    records = load_demo_records(csv_path=str(missing_path), limit=None)

    assert len(records) == 3
    assert records[0].track_name == "Sample Pop Signal"
