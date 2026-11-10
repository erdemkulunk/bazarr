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


@pytest.mark.parametrize('failure', ['permission', 'unsupported', 'existing', 'legacy', 'empty', 'skipped'])
def test_native_queued_upload_reports_real_create_failures(tmp_path, monkeypatch, failure):
    """Run the real enqueue/consumer path, preserving collision indexing and the legacy failure contract."""
    import errno
    from io import BytesIO
    from types import SimpleNamespace
    import app.jobs_queue as queue_module
    from bazarr.subtitles import upload

    queue = queue_module.JobsQueue()
    events, indexed, history = [], [], []
    monkeypatch.setattr(upload, 'jobs_queue', queue)
    monkeypatch.setattr(queue_module, 'event_stream', lambda **event: events.append(event))
    metadata = SimpleNamespace(profileId=1, sonarrSeriesId=1, sonarrEpisodeId=1,
                               season=1, episode=1, imdbId='tt999000169', tvdbId=999000169)
    monkeypatch.setattr(upload, 'database', SimpleNamespace(
        execute=lambda query: SimpleNamespace(first=lambda: metadata)))
    monkeypatch.setattr(upload, 'get_profiles_list', lambda profile: {'originalFormat': False})
    monkeypatch.setattr(upload, 'store_subtitles', lambda episode: indexed.append(episode))
    monkeypatch.setattr(upload, 'history_log', lambda *args, **kwargs: history.append(args))
    video, target = tmp_path / 'episode.mkv', tmp_path / 'episode.en.srt'
    if failure in ['existing', 'empty']:
        target.write_bytes(b'Concurrent manual winner.\n')
    elif failure == 'skipped':
        monkeypatch.setattr(Subtitle, 'text', property(lambda self: None))
    elif failure == 'legacy':
        def fail_save(*args, **kwargs):
            raise PermissionError(errno.EACCES, 'simulated legacy write permission failure')
        monkeypatch.setattr(upload, 'save_subtitles', fail_save)
    elif failure != 'empty':
        def fail_link(source, destination):
            assert open(source, 'rb').read() == SRT
            raise OSError(errno.EPERM if failure == 'permission' else errno.EOPNOTSUPP,
                          'simulated hardlink publication failure')
        monkeypatch.setattr(publication.os, 'link', fail_link)

    if failure == 'empty':
        target.chmod(0o600)
        monkeypatch.setitem(upload.settings.general, 'chmod_enabled', True)
        monkeypatch.setitem(upload.settings.general, 'chmod', '640')
        monkeypatch.setattr(Subtitle, 'get_modified_content', lambda *args, **kwargs: b'')

    job_id = upload.manual_upload_subtitle(
        path=str(video), language='en', forced=False, hi=False, media_type='series',
        subtitle=BytesIO(SRT), filename='episode.en.srt', audio_language='[]',
        sonarrSeriesId=1, sonarrEpisodeId=1, create_only=failure != 'legacy')
    assert queue.get_job_status(job_id) == 'pending'
    assert queue.jobs_pending_queue[0].kwargs['create_only'] is (failure != 'legacy')
    succeeded = queue._run_job()
    failed = failure in ['permission', 'unsupported', 'empty', 'skipped']
    if failure == 'empty':
        assert target.read_bytes() == b'Concurrent manual winner.\n'
        assert target.stat().st_mode & 0o777 == 0o600
    assert succeeded is not failed
    assert queue.get_job_status(job_id) == ('failed' if failed else 'completed')
    assert events[-1]['payload']['status'] == ('failed' if failed else 'completed')
    assert indexed == ([1] if failure == 'existing' else [])
    assert history == []
    assert not list(tmp_path.glob('.bazarr-*.tmp'))
    if failure in ['existing', 'empty']:
        assert target.read_bytes() == b'Concurrent manual winner.\n'
    else:
        assert not target.exists()


def test_create_only_permissions_apply_before_publication(tmp_path, monkeypatch):
    target = tmp_path / 'episode.en.srt'
    link = os.link

    def check_prepared_inode(source, destination):
        assert open(source, 'rb').read() == SRT
        assert os.stat(source).st_mode & 0o777 == 0o640
        assert not target.exists()
        return link(source, destination)

    monkeypatch.setattr(publication.os, 'link', check_prepared_inode)
    publication.create_file(str(target), SRT, chmod=0o640)
    assert target.read_bytes() == SRT
    assert target.stat().st_mode & 0o777 == 0o640
