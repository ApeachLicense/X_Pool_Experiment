import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from analyze_retrieval import query_records, run
from modules.metrics import t2v_metrics


class RetrievalAnalysisTests(unittest.TestCase):
    def test_existing_run_report_is_preserved_before_evaluation(self):
        root = Path(__file__).resolve().parents[1] / 'outputs'
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='analysis_test_', dir=root) as directory:
            folder = Path(directory).resolve()
            self.assertTrue(folder.is_relative_to(root.resolve()))
            original = '{"status": "finished", "evidence": "preserve"}\n'
            (folder / 'run_summary.json').write_text(original, encoding='utf-8')
            with self.assertRaises(FileExistsError):
                run(SimpleNamespace(source_root=folder, results_dir=folder))
            self.assertEqual((folder / 'run_summary.json').read_text(encoding='utf-8'), original)

    def test_query_video_axes_and_rank_convention(self):
        scores = torch.tensor([[0.9, 0.2, 0.1], [0.8, 0.7, 0.1], [0.2, 0.9, 0.1]])
        queries = [{'video_id': str(i), 'sentence': 'query ' + str(i)} for i in range(3)]
        logits = torch.arange(36).float().reshape(3, 1, 4, 3) / 20
        attention = logits.softmax(dim=2)
        rows = query_records(scores, queries, attention)
        self.assertEqual([r['rank'] for r in rows], [1, 2, 3])
        self.assertEqual([r['top1_video_id'] for r in rows], ['0', '0', '1'])
        self.assertEqual(rows[2]['top1_caption'], 'query 1')
        self.assertTrue(torch.allclose(torch.tensor(json.loads(rows[2]['gt_attention'])), attention[2, 0, :, 2]))
        self.assertTrue(torch.allclose(torch.tensor(json.loads(rows[2]['top1_attention'])), attention[1, 0, :, 2]))
        metrics = t2v_metrics(scores.unsqueeze(1))
        self.assertEqual(metrics['R1'], 100 / 3)
        self.assertEqual(metrics['MeanR'], 2)

    def test_ties_match_official_metric_and_attention_bounds(self):
        scores = torch.ones(4, 4)
        queries = [{'video_id': str(i), 'sentence': str(i)} for i in range(4)]
        attention = torch.full((4, 2, 12, 4), 1 / 12)
        rows = query_records(scores, queries, attention)
        metric = t2v_metrics(scores.unsqueeze(1))
        self.assertEqual(100 * sum(r['rank'] == 1 for r in rows) / 4, metric['R1'])
        self.assertEqual(sum(r['rank'] for r in rows) / 4, metric['MeanR'])
        self.assertTrue(all(r['gt_score_ties'] == 4 for r in rows))
        self.assertTrue(all(abs(r['gt_attention_entropy'] - 1) < 1e-6 for r in rows))


if __name__ == '__main__':
    unittest.main()
