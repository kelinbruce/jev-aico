"""Validate v3 evidence certificates before the existing candidate-loss trainer."""
import json
from collections import defaultdict
from pathlib import Path

from nimble.datasets.contrastive_data import fingerprint
from nimble.datasets.scaled_evidence import audit_release, build_group, make_rows


def validate_evidence_data(raw, manifest):
    if manifest.get('pipeline_version') != 'evidence-curation-v3':
        raise ValueError('Unsupported evidence dataset version')
    if 'source_records' in manifest:
        # Provenance only: never appended to the training or evaluation rows.
        sources = manifest['source_records']
    else:
        source_path = Path(manifest['source'])
        # GPU copies can have a different project root.
        if not source_path.exists():
            from nimble.paths import PROJECT_ROOT
            source_path = PROJECT_ROOT / 'data' / source_path.parent.name / source_path.name
        sources = [json.loads(l) for l in source_path.read_text().splitlines() if l.strip()]
    if fingerprint(sources) != manifest['source_sha256']:
        raise ValueError('Evidence source fingerprint changed')
    if audit_release(raw, sources) != manifest['training']:
        raise ValueError('Evidence coverage audit differs from manifest')
    source_map = {s['id']: s for s in sources}
    groups = defaultdict(list)
    for row in raw:
        groups[row['family']].append(row)
    for gid, rows in groups.items():
        row = rows[0]
        cert = row['evidence_certificate']
        spec = cert['spec']
        group = build_group(gid, source_map[row['provenance']['source_id']], spec,
                            {'base_state_json': spec['base_state_json'], 'focus_evidence': spec['focus_evidence']},
                            cert['verified_pair'])
        reconstructed = make_rows(group, cert['rule_audit'], cert['verified_pair'], cert['pair_fact_states'],
                                  cert['context_audit'], cert['full_context_fact_states'])
        if sorted(rows, key=lambda r: r['id']) != sorted(reconstructed, key=lambda r: r['id']):
            raise ValueError('Evidence certificate/context reconstruction differs: ' + gid)
    return {gid: [r['variant'] for r in rows] for gid, rows in groups.items()}
