"""CPU-prepared training-graph identities, separate from predictor outputs.

No source parsing or graph capture occurs on the scheduling path. Features are
descriptive; their distance is a transfer hypothesis, not an interference law.
"""
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
import math

SCHEMA = 'fifo_training_fingerprint_v1'


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()).hexdigest()


def histogram(values):
    counts = Counter(values)
    total = sum(counts.values())
    return {str(k): v / total for k, v in sorted(counts.items())} if total else {}


def graph_summary(graph):
    """Bind the real captured forward/loss/backward/optimizer graph and shapes."""
    if (not graph.training_mode or graph.coverage.backward_capture_quality != 'strict'
            or not any(n.phase == 'backward' for n in graph.nodes)):
        raise ValueError('strict complete training graph required')
    nodes = [asdict(n) for n in graph.nodes]
    edges = [asdict(e) for e in graph.tensor_edges]
    if not nodes or not edges:
        raise ValueError('empty training graph')
    topology = [{k: n[k] for k in ('node_id', 'canonical_op_id', 'phase', 'depth',
                'fan_in', 'fan_out', 'input_numel', 'output_numel', 'parameter_bytes')}
                for n in nodes]
    edge_identity = [{k: e[k] for k in ('producer_node_id', 'consumer_node_id',
        'producer_output_index', 'consumer_input_index', 'tensor_role', 'shape',
        'dtype', 'stride', 'memory_format', 'is_view')} for e in edges]
    widths = Counter(n['depth'] for n in nodes)
    if any(not isinstance(e['numel'], int) or e['numel'] < 0 for e in edges):
        raise ValueError('unknown tensor shapes cannot establish similarity')
    return {'graph_sha256': digest({'nodes': topology, 'edges': edge_identity}),
        'operations': histogram(n['canonical_op_id'] for n in nodes),
        'phases': histogram(n['phase'] for n in nodes),
        'size_bins': histogram(int(math.log2(max(1, e['numel']))) for e in edges),
        'structure': [len(nodes), len(edges), max(widths), max(widths.values())],
        # Sums are graph-incidence features, not unique allocated-memory labels.
        'sizes': [sum(n['parameter_bytes'] for n in nodes),
                  sum(e['tensor_bytes'] or 0 for e in edges if e['tensor_role'] == 'activation')],
        'training_graph': True, 'feature_memory_semantics': 'graph_incidence_bytes'}


def make_fingerprint(summary, recipe, *, hardware, runtime, artifact, source):
    required = ('batch', 'accumulation', 'precision', 'optimizer', 'data_path',
                'backend', 'ballast_mib')
    if any(k not in recipe for k in required) or not all((hardware, runtime, artifact, source)):
        raise ValueError('incomplete fingerprint identity')
    if not summary.get('training_graph') or not summary.get('graph_sha256'):
        raise ValueError('missing captured graph')
    compatibility = {k: recipe[k] for k in ('accumulation', 'precision', 'optimizer', 'data_path', 'backend')}
    compatibility.update(schema=SCHEMA, hardware=hardware, runtime=runtime, artifact=artifact)
    exact = digest({'graph': summary['graph_sha256'], 'recipe': recipe,
                    'compatibility': compatibility})
    return {'schema': SCHEMA, 'exact': exact, 'compatibility': compatibility,
            'recipe': recipe, 'graph': summary, 'source_provenance': source}


def log_distance(a, b):
    if len(a) != len(b) or not a:
        raise ValueError('incompatible vector')
    values = []
    for x, y in zip(a, b):
        if (not isinstance(x, (int, float)) or isinstance(x, bool)
                or not isinstance(y, (int, float)) or isinstance(y, bool)
                or not math.isfinite(x + y) or min(x, y) < 0):
            raise ValueError('invalid structural feature')
        values.append(min(1., abs(math.log1p(x) - math.log1p(y)) / math.log(4.)))
    return sum(values) / len(values)


def histogram_distance(a, b):
    return min(1., sum(abs(a.get(k, 0.) - b.get(k, 0.)) for k in a.keys() | b.keys()) / 2.)


def structural_distances(a, b):
    if a['compatibility'] != b['compatibility']:
        raise ValueError('incompatible hardware/runtime/recipe')
    x, y = a['graph'], b['graph']
    graph = (histogram_distance(x['operations'], y['operations']) +
             histogram_distance(x['phases'], y['phases']) +
             log_distance(x['structure'], y['structure'])) / 3.
    shape = (histogram_distance(x['size_bins'], y['size_bins']) +
             log_distance(x['sizes'] + [a['recipe']['batch'], a['recipe']['ballast_mib']],
                          y['sizes'] + [b['recipe']['batch'], b['recipe']['ballast_mib']])) / 2.
    return graph, shape


def context(fingerprints, *, resident_count, starting=0, finishing=0, phase_peers=None):
    """Target is excluded by caller; peer vector has no job-id dependence."""
    if resident_count != len(fingerprints) + 1 + starting + finishing:
        raise ValueError('peer membership/count mismatch')
    phases = {'resident': resident_count, 'training': len(fingerprints) + 1,
              'starting': starting, 'finishing': finishing}
    ops, sizes, structures = {}, [], []
    for fp in fingerprints:
        for k, v in fp['graph']['operations'].items():
            ops[k] = ops.get(k, 0.) + v / max(1, len(fingerprints))
        sizes.append(fp['graph']['sizes'] + [fp['recipe']['batch'], fp['recipe']['ballast_mib']])
        structures.append(fp['graph']['structure'])
    def moments(rows, n):
        if not rows: return [0.] * (2 * n)
        means = [sum(row[i] for row in rows) / len(rows) for i in range(n)]
        spreads = [sum((row[i] - means[i]) ** 2 for row in rows) ** .5 / len(rows) for i in range(n)]
        return means + spreads
    result = {'phases': phases, 'exact_peers': sorted(fp['exact'] for fp in fingerprints),
              'operations': ops, 'moments': moments(sizes, 4) + moments(structures, 4)}
    if phase_peers is not None:
        if set(phase_peers) != {'starting', 'finishing'} or any(
                len(phase_peers[k]) != phases[k] for k in phase_peers):
            raise ValueError('non-training peer identity/count mismatch')
        result['phase_peers'] = {k: sorted(fp['exact'] for fp in v) for k, v in phase_peers.items()}
    result['key'] = digest(result)
    return result


def peer_distance(a, b):
    # Do not borrow evidence across a different occupancy or phase composition.
    if a['phases'] != b['phases'] or a.get('phase_peers') != b.get('phase_peers'):
        return 1.
    return (histogram_distance(a['operations'], b['operations']) +
            log_distance(a['moments'], b['moments'])) / 2.
