import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from scripts.download_dataset import extract_verified
from scripts.judge import judge_rq1, judge_rq2_ablation, judge_rq3
from puzzles.common.types import Pos, Direction

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/model_runs'))
from run_openai_compatible import make_cache_key


def test_mosaic_paint_blocks_return():
    from puzzles.mosaic_grid.engine import MosaicGrid, MosaicState, apply_action, get_legal_actions
    grid = MosaicGrid(3, 1, frozenset(), {}, {}, Pos(0, 0), Pos(0, 2))
    state = apply_action(grid, MosaicState(Pos(0, 0), 0, 0), Direction.RIGHT)
    assert state.agent_pos == Pos(0, 1)
    assert Direction.LEFT not in get_legal_actions(grid, state)


def test_chain_door_consumes_key_without_moving():
    from puzzles.chain_vault.engine import ChainVaultGrid, ChainVaultState, DoorInfo, ChainVaultAction, apply_action
    grid = ChainVaultGrid(3, 1, frozenset(), {}, {'d': DoorInfo('d', Pos(0, 1), ('red',))}, Pos(0, 0), Pos(0, 2))
    state = ChainVaultState(Pos(0, 0), (('red', 1),), frozenset(), frozenset())
    opened = apply_action(grid, state, ChainVaultAction.RIGHT)
    assert opened.agent_pos == state.agent_pos and opened.inventory == ()
    assert opened.opened_doors == frozenset({'d'})
    assert apply_action(grid, opened, ChainVaultAction.RIGHT).agent_pos == Pos(0, 1)


def test_tide_wait_uses_oxygen_and_changes_tidal_window():
    from puzzles.tide_grid.engine import TideGridGrid, TidalZone, TideGridAction, make_initial_state, apply_action, get_legal_actions
    grid = TideGridGrid(3, 1, frozenset(), {'A': TidalZone('A', frozenset({Pos(0, 1)}), 1, 1, 0)}, {}, Pos(0, 0), Pos(0, 2), 2)
    state = make_initial_state(grid)
    assert TideGridAction.RIGHT not in get_legal_actions(grid, state)
    waited = apply_action(grid, state, TideGridAction.WAIT)
    assert waited.time == 1 and TideGridAction.RIGHT in get_legal_actions(grid, waited)
    exhausted = apply_action(grid, waited, TideGridAction.WAIT)
    assert get_legal_actions(grid, exhausted) == []


@pytest.mark.parametrize('judge,truth,correct,wrong', [
    (judge_rq1, {'agent_pos': [2, 3]}, 'POSITION: (2, 3)', 'POSITION: (3, 2)'),
    (judge_rq2_ablation, {'answer': 'NO'}, 'SOLVABLE: NO', 'SOLVABLE: YES'),
    (judge_rq3, {'fatal_step_1indexed': 4}, 'FATAL STEP: 4', 'FATAL STEP: 3'),
])
def test_deterministic_judges(judge, truth, correct, wrong):
    assert judge(truth, correct)['correct']
    assert not judge(truth, wrong)['correct']
    assert judge(truth, '')['parse_fail']


def test_cache_distinguishes_generation_settings():
    options = dict(base_url='https://example.invalid/v1', model_name='test', prompt='p', images_b64=['a'])
    baseline = make_cache_key(**options)
    assert baseline != make_cache_key(**options, temperature=0.8)
    assert baseline != make_cache_key(**options, max_tokens=100)
    assert baseline != make_cache_key(**options, presence_penalty=0.5)


def test_archive_checksums_and_nonempty_destination(tmp_path):
    archive = tmp_path / 'data.zip'
    name, content = 'dataset/example.json', b'[]'
    manifest = {'files': {name: hashlib.sha256(content).hexdigest()}}
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr(name, content)
    destination = tmp_path / 'out'
    assert extract_verified(archive, destination, manifest) == 1
    assert (destination / 'example.json').read_bytes() == content
    with pytest.raises(FileExistsError):
        extract_verified(archive, destination, manifest)
    manifest['files'][name] = 'bad'
    with pytest.raises(ValueError, match='Checksum'):
        extract_verified(archive, tmp_path / 'bad', manifest)
    assert not (tmp_path / 'bad').exists()


def test_archive_rejects_extra_files(tmp_path):
    archive = tmp_path / 'data.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('../escape', 'bad')
    with pytest.raises(ValueError):
        extract_verified(archive, tmp_path / 'out', {'files': {}})


@pytest.mark.parametrize('script', ['run_openai_compatible.sh', 'run_openai_compatible_models.sh'])
def test_shell_launchers_locate_repository_environment(script):
    result = subprocess.run(['bash', str(ROOT / 'scripts/model_runs' / script), '--help'], cwd='/tmp', capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--dataset-file' in result.stdout


def test_dry_run_does_not_delete_output_or_create_cache(tmp_path):
    from PIL import Image
    image = tmp_path / 'image.png'
    Image.new('RGB', (2, 2)).save(image)
    dataset = tmp_path / 'data.json'
    dataset.write_text(json.dumps([{'eval_id': 'one', 'prompt': 'test', 'image_path': 'image.png'}]))
    output = tmp_path / 'response.jsonl'
    output.write_text('existing response\n')
    cache = tmp_path / 'cache'
    command = [sys.executable, str(ROOT / 'scripts/model_runs/run_openai_compatible.py'), '--dataset-file', str(dataset), '--model', 'test', '--base-url', 'https://example.invalid/v1', '--output', str(output), '--cache-dir', str(cache), '--overwrite', '--dry-run']
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output.read_text() == 'existing response\n'
    assert not cache.exists()
