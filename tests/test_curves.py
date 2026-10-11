from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from plot_curves import parse_csv, parse_log, plot_folds


class CurvesTests(unittest.TestCase):
    def test_csv_loss_only_and_sorted_epochs(self):
        rows = parse_csv('epoch,loss\n2,1e-2\n1,+.03\n')[0]
        self.assertEqual([row['epoch'] for row in rows], [1, 2])
        self.assertEqual(rows[1]['loss'], 0.01)

    def test_log_whitespace_and_exponent(self):
        text = 'Epoch 1 | uni = +1e-05, multi= .2, mmc = 0.3 total_loss = 0.50001 || Acc= .9 AUC(macro)= 0.85 F1(macro) = .8\nfold 7, test_loss=0.1'
        rows = parse_log(text)[7]
        self.assertEqual(rows[0]['uni'], 1e-5)
        self.assertEqual(rows[0]['F1(macro)'], 0.8)

    def test_fold_header_and_trailer(self):
        folds = parse_log('Fold 2\nEpoch 1 | loss=0.9\nEpoch 2 | loss=0.8\nfold 2, test_loss=0.7\nEpoch 1 | loss=0.6\nfold 8, test_loss=0.5')
        self.assertEqual(set(folds), {2, 8})
        self.assertEqual(len(folds[2]), 2)

    def test_inconsistent_fold_ids_are_errors(self):
        with self.assertRaisesRegex(ValueError, 'disagree'):
            parse_log('Fold 2\nEpoch 1 loss=0.3\nfold 7, test_loss=0.1')

    def test_invalid_rows_never_disappear_silently(self):
        for text in ['Epoch X | loss=1', 'Epoch 1 | loss=NaN', 'Epoch 1 | loss=1e999', 'Epoch 1 | loss=oops']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_log(text)
        for csv in ['epoch,loss\n1,nan\n', 'epoch,loss\n1,2,3\n', 'epoch,loss\n1\n', 'epoch,loss,loss\n1,2,3\n']:
            with self.subTest(csv=csv), self.assertRaises(ValueError):
                parse_csv(csv)

    def test_duplicates_and_inconsistent_metrics_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            parse_csv('epoch,loss\n1,.2\n1,.1\n')
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            parse_log('Epoch 1 loss=.2 Acc=.8\nEpoch 2 loss=.1')

    def test_headless_export_uses_actual_data_and_protects_existing_files(self):
        import matplotlib.axes
        real_plot = matplotlib.axes.Axes.plot
        captured = []

        def capture(axis, x, y, *args, **kwargs):
            captured.append((list(x), list(y), kwargs['label']))
            return real_plot(axis, x, y, *args, **kwargs)

        folds = parse_csv('fold,epoch,train_loss,val_loss,Acc\n4,2,.3,.4,.9\n4,1,.6,.7,.8\n')
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(matplotlib.axes.Axes, 'plot', capture):
                paths = plot_folds(folds, Path(folder))
            self.assertEqual(paths[0].name, 'fold_4_curves.png')
            self.assertEqual(paths[0].read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            self.assertIn(([1, 2], [.6, .3], 'train_loss'), captured)
            original = paths[0].read_bytes()
            with self.assertRaises(FileExistsError):
                plot_folds(folds, Path(folder))
            self.assertEqual(paths[0].read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
