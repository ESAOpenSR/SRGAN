#!/usr/bin/env python3
"""Prepare and atomically publish the versioned model Hub layout.

Run without --publish to inspect the staged cards, examples and checksums.
Existing URLs remain available: source files are copied, never deleted. Large
checkpoint/sample copies are performed by the Hub without downloading weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from huggingface_hub import CommitOperationAdd, CommitOperationCopy, HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', default='simon-donike/SR-GAN')
    parser.add_argument('--stage', type=Path, help='optional staging directory; defaults to a fresh private temporary directory')
    parser.add_argument('--cache-dir', type=Path, default=Path('/tmp/srgan-hf-repackage-cache'))
    parser.add_argument('--publish', action='store_true', help='create one atomic Hub commit after preparation')
    args = parser.parse_args()
    source = ROOT / 'hub'
    stage = args.stage.absolute() if args.stage is not None else Path(tempfile.mkdtemp(prefix='srgan-hf-structured-'))
    if any(path.is_symlink() for path in (stage, *stage.parents)):
        parser.error('--stage and its parents must not be symlinks')
    if stage == source.resolve() or source.resolve() in stage.parents or stage in source.resolve().parents:
        parser.error('--stage must be separate from the Hub source directory')
    catalog = json.loads((source / 'catalog.json').read_text())
    compatibility = json.loads((source / 'compatibility.json').read_text())
    source_revision = compatibility['source_revision']
    aliases = compatibility['aliases']
    if len(set(aliases.values())) != len(aliases):
        raise ValueError('Canonical copy destinations must be unique')
    api = HfApi()
    parent_commit = api.model_info(args.repo_id).sha
    inventory = api.model_info(args.repo_id, revision=source_revision, files_metadata=True)
    remote = {entry.rfilename: entry for entry in inventory.siblings}
    stage.mkdir(parents=True, exist_ok=True, mode=0o700)
    stage.chmod(0o700)
    if any(path.is_symlink() for path in stage.rglob('*')):
        parser.error('--stage must not contain symlinks')
    # A stage can be reused without silently carrying removed cards/examples.
    # Clean only files recorded by a previous invocation of this script.
    marker = stage / '.generated-files.json'
    if marker.is_file():
        for relative in json.loads(marker.read_text()):
            previous = (stage / relative).resolve()
            if stage not in previous.parents:
                raise ValueError('Invalid generated-file marker path')
            if previous.is_file():
                previous.unlink()
    generated: list[str] = []
    local_paths: dict[str, Path] = {}
    for path in sorted(source.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        relative = path.relative_to(source).as_posix()
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        local_paths[relative] = target
        generated.append(relative)
    artifact_metadata: dict[str, dict] = {}
    copies = []
    for old_path, canonical_path in aliases.items():
        if old_path not in remote:
            raise ValueError(f'Missing source artifact: {old_path}')
        if canonical_path in local_paths:
            raise ValueError(f'Copy conflicts with authored content: {canonical_path}')
        entry = remote[old_path]
        if entry.lfs is not None:
            digest = entry.lfs.sha256
        else:
            cached = hf_hub_download(args.repo_id, old_path, revision=source_revision, cache_dir=args.cache_dir)
            digest = sha256(Path(cached))
        artifact_metadata[canonical_path] = {
            'bytes': entry.size,
            'sha256': digest,
            'source_path': old_path,
            'source_revision': source_revision,
        }
        copies.append(CommitOperationCopy(src_path_in_repo=old_path, path_in_repo=canonical_path, src_revision=source_revision))
    for relative, path in local_paths.items():
        artifact_metadata[relative] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    required = {'README.md', 'LICENSE', 'NOTICE.md', 'catalog.json', 'compatibility.json'}
    required.update(catalog['examples'].values())
    required.update((catalog['samples']['rgb_nir_batch'], catalog['samples']['dataset_archive']))
    for model in catalog['models']:
        required.update((model['config'], model['inference_checkpoint'], model['directory'] + '/README.md', model['directory'] + '/LICENSE'))
        for field in ('training_checkpoint', 'training_config', 'split'):
            if field in model:
                required.add(model[field])
    missing = required - artifact_metadata.keys()
    if missing:
        raise ValueError('Incomplete release: ' + ', '.join(sorted(missing)))
    for model in catalog['models']:
        directory = model['directory']
        prefix = directory + '/'
        files = {key[len(prefix):]: value for key, value in sorted(artifact_metadata.items()) if key.startswith(prefix)}
        manifest = {
            'schema_version': 1,
            'model': model,
            'license': 'MIT',
            'source_revision': source_revision,
            'checkpoint_bytes_unchanged': True,
            'files': files,
        }
        manifest_path = stage / directory / 'manifest.json'
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        manifest_name = manifest_path.relative_to(stage).as_posix()
        local_paths[manifest_name] = manifest_path
        generated.append(manifest_name)
        checksums = dict(files)
        checksums['manifest.json'] = {'sha256': sha256(manifest_path)}
        checksum_path = stage / directory / 'SHA256SUMS'
        checksum_path.write_text(''.join(f'{value["sha256"]}  {name}\n' for name, value in sorted(checksums.items())))
        checksum_name = checksum_path.relative_to(stage).as_posix()
        local_paths[checksum_name] = checksum_path
        generated.append(checksum_name)
    release = {
        'schema_version': 1,
        'source_revision': source_revision,
        'preserve_existing_artifact_downloads': True,
        'updated_existing_files': ['README.md'],
        'models': [model['directory'] for model in catalog['models']],
        'compatibility_aliases': len(aliases),
        'files': {},
    }
    for relative, path in local_paths.items():
        artifact_metadata[relative] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    release['files'] = dict(sorted(artifact_metadata.items()))
    release_path = stage / 'release.json'
    release_path.write_text(json.dumps(release, indent=2) + '\n')
    local_paths['release.json'] = release_path
    generated.append('release.json')
    marker.write_text(json.dumps(generated, indent=2) + '\n')
    operations = copies + [CommitOperationAdd(path_in_repo=name, path_or_fileobj=path) for name, path in sorted(local_paths.items())]
    summary = {'repo': args.repo_id, 'source_revision': source_revision, 'parent_commit': parent_commit, 'server_side_copies': len(copies), 'authored_files': len(local_paths), 'stage': str(stage), 'published': False}
    if args.publish:
        commit = api.create_commit(
            repo_id=args.repo_id,
            operations=operations,
            parent_commit=parent_commit,
            commit_message='Organize versioned models, MIT model cards, examples and compatible downloads',
        )
        summary.update(published=True, commit=commit.oid, url=commit.commit_url)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
