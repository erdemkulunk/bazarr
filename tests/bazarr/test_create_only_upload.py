import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from subliminal_patch import publication
from subliminal_patch.core import save_subtitles
from subliminal_patch.subtitle import Subtitle
from subzero.language import Language

SRT = b'1\n00:00:00,100 --> 00:00:01,000\nCreated English.\n\n'


def test_native_create_only_preserves_existing_and_legacy_replace(tmp_path):
    video = tmp_path / 'episode.mkv'
    target = tmp_path / 'episode.en.srt'
    subtitle = Subtitle(Language('eng'))
    subtitle.content = SRT
    assert subtitle.is_valid()
    target.write_bytes(b'Existing manual English.\n')
    with pytest.raises(FileExistsError):
        save_subtitles(str(video), [subtitle], overwrite=False)
    assert target.read_bytes() == b'Existing manual English.\n'
    save_subtitles(str(video), [subtitle])
    assert target.read_bytes() == SRT
    assert not list(tmp_path.glob('.bazarr-*.tmp'))


def test_competing_manual_file_wins_at_publication_boundary(tmp_path, monkeypatch):
    target = tmp_path / 'episode.en.srt'
    rival = b'Fresh manual subtitle discovered while generation was running.\n'
    link = os.link

    def publish_after_manual_write(source, destination):
        assert open(source, 'rb').read() == SRT
        assert not target.exists()
        target.write_bytes(rival)
        return link(source, destination)

    monkeypatch.setattr(publication.os, 'link', publish_after_manual_write)
    with pytest.raises(FileExistsError):
        publication.create_file(str(target), SRT)
    assert target.read_bytes() == rival
    assert not list(tmp_path.glob('.bazarr-*.tmp'))


def test_concurrent_create_only_writers_publish_one_complete_file(tmp_path):
    target = tmp_path / 'episode.en.srt'
    candidates = [SRT + str(i).encode() for i in range(8)]

    def write(content):
        try:
            publication.create_file(str(target), content)
            return True
        except FileExistsError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(write, candidates)) == 1
    assert target.read_bytes() in candidates
    assert not list(tmp_path.glob('.bazarr-*.tmp'))


def test_failed_write_leaves_no_published_or_temporary_file(tmp_path, monkeypatch):
    target = tmp_path / 'episode.en.srt'

    def fail(_):
        raise OSError('simulated fsync failure')

    monkeypatch.setattr(publication.os, 'fsync', fail)
    with pytest.raises(OSError):
        publication.create_file(str(target), SRT)
    assert not target.exists()
    assert not list(tmp_path.glob('.bazarr-*.tmp'))
