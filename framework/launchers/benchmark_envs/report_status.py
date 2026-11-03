"""Report only observed installation/check results; do not infer job success."""
from datetime import datetime,timezone
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ASSETS=Path('/data/fd/donghaosen/Self-Distillation-AgentRL')
LOGS=ASSETS/'logs'
install_codes={'alfworld':'alfworld-install-isolated.exit','webshop':'webshop-install-isolated.exit','searchqa':'searchqa-install-repaired.exit','sciworld':'sciworld-install-isolated.exit'}
report={'updated_at':datetime.now(timezone.utc).isoformat(),'asset_root':str(ASSETS),'training_started':False,'environments':{}}
for bench,code in install_codes.items():
 p=LOGS/code
 entry={'conda_name':'agentrl-'+bench,'prefix':str(ROOT/'local_assets/conda/envs'/('agentrl-'+bench)),
        'installation_exit_code':int(p.read_text().strip()) if p.exists() else None,
        'smoke':json.loads((LOGS/(bench+'-smoke.json')).read_text()) if (LOGS/(bench+'-smoke.json')).exists() else None}
 adapter=LOGS/(bench+'-adapter-check.json')
 entry['native_adapter_check']=json.loads(adapter.read_text()) if adapter.exists() else None
 entry['conda_snapshot']=str(ROOT/'launchers/benchmark_envs/locks'/bench/'environment.yml')
 entry['pip_snapshot']=str(LOGS/(bench+'-final-pip-freeze.txt'))
 report['environments'][bench]=entry
full=LOGS/'searchqa-full-index-check.json'
report['full_search_index_check']=json.loads(full.read_text()) if full.exists() else None
report['all_installed']=all(v['installation_exit_code']==0 for v in report['environments'].values())
report['all_smoke_checks_completed']=all(v['smoke'] is not None for v in report['environments'].values())
report['all_native_adapter_imports_passed']=all(v['native_adapter_check'] and v['native_adapter_check']['import_passed'] for v in report['environments'].values())
report['all_snapshots_present']=all(Path(v['conda_snapshot']).is_file() and Path(v['pip_snapshot']).is_file() for v in report['environments'].values())
report['all_ready']=(report['all_installed'] and report['all_smoke_checks_completed'] and report['all_native_adapter_imports_passed'] and report['all_snapshots_present'] and report['full_search_index_check'] is not None)

(LOGS/'environment-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'all_installed':report['all_installed'],'all_smoke_checks_completed':report['all_smoke_checks_completed'],'full_search_index_checked':full.exists(),'smoke_results':{k:v['smoke'] is not None for k,v in report['environments'].items()}},indent=2))
