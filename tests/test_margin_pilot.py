from pathlib import Path
import tempfile
import unittest

import torch

from margin_pilot import make_records
from modules.loss import CLIPLoss, MarginCLIPLoss
from test_training_resume import make_trainer


class MarginPilotTests(unittest.TestCase):
    def test_zero_margin_preserves_original_loss_and_gradients_exactly(self):
        scores=torch.tensor([[.5,.3],[.2,.6]],dtype=torch.float64,requires_grad=True)
        scale=torch.tensor(2.0,dtype=torch.float64,requires_grad=True)
        original=CLIPLoss()(scores,scale)
        a=torch.autograd.grad(original,(scores,scale),retain_graph=True)
        zero=MarginCLIPLoss(0)(scores,scale)
        b=torch.autograd.grad(zero,(scores,scale))
        self.assertTrue(torch.equal(original,zero))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)))

    def test_margin_increases_pressure_on_confusing_positive_negative_pairs(self):
        scores=torch.tensor([[.50,.49],[.48,.50]],requires_grad=True)
        before=scores.detach().clone();scale=torch.tensor(2.0)
        baseline=CLIPLoss()(scores,scale)
        base_grad=torch.autograd.grad(baseline,scores,retain_graph=True)[0]
        candidate=MarginCLIPLoss(.02)(scores,scale)
        grad=torch.autograd.grad(candidate,scores)[0]
        self.assertGreater(float(candidate),float(baseline))
        self.assertTrue(bool((torch.diag(grad)<torch.diag(base_grad)).all()))
        self.assertGreater(float(grad[0,1]),float(base_grad[0,1]))
        self.assertTrue(torch.equal(scores.detach(),before))
        self.assertTrue(bool(torch.isfinite(grad).all()))

    def test_invalid_margin_and_non_square_correspondence_are_rejected(self):
        for margin in [-.1,float('nan'),float('inf'),1]:
            with self.assertRaises(ValueError):MarginCLIPLoss(margin)
        with self.assertRaises(ValueError):MarginCLIPLoss(.02)(torch.ones(2,3),torch.tensor(1.))

    def test_internal_split_is_deterministic_video_disjoint_and_complete(self):
        ids=[f'video{i}' for i in range(12)]
        annotations=[{'video_id':v,'caption':f'{v} caption {j}'} for v in ids for j in range(2)]
        a,b=make_records(ids,annotations,3,24)
        aa,bb=make_records(ids,annotations,3,24)
        self.assertEqual((a,b),(aa,bb))
        train={r['video_id'] for r in a};valid={r['video_id'] for r in b}
        self.assertFalse(train&valid);self.assertEqual(train|valid,set(ids))
        self.assertEqual(len(a),18);self.assertEqual(len(b),3)
        self.assertEqual([r['pair_id'] for r in a],list(range(18)))

    def test_repeated_training_chunks_preserve_continuous_model_and_optimizer(self):
        root=Path(__file__).resolve().parents[1]/'outputs';root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='margin_test_',dir=root) as directory:
            folder=Path(directory).resolve();self.assertTrue(folder.is_relative_to(root.resolve()))
            full=make_trainer(folder/'full',6,workers=0);full.train()
            chunked=make_trainer(folder/'chunks',2,workers=0)
            for _ in range(3):chunked.train()
            self.assertEqual(full.global_step,6);self.assertEqual(chunked.global_step,6)
            for name,value in full.model.state_dict().items():self.assertTrue(torch.equal(value,chunked.model.state_dict()[name]),name)
            left=full.optimizer.state_dict()['state'];right=chunked.optimizer.state_dict()['state']
            for key,values in left.items():
                for name,value in values.items():
                    if isinstance(value,torch.Tensor):self.assertTrue(torch.equal(value,right[key][name]))
                    else:self.assertEqual(value,right[key][name])
            self.assertEqual(full.lr_scheduler.state_dict(),chunked.lr_scheduler.state_dict())


if __name__=='__main__':unittest.main()
